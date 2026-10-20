"""The capture path, crops_from_video, on synthetic videos in each container the app
receives: OpenCV-written AVI and MP4, and WebM written the way browser MediaRecorder
writes it."""
import itertools

import cv2
import numpy as np
import pytest

from app.live_vitals import config
from app.live_vitals.capture import session
from app.live_vitals.preprocess.frames import crop_resize
from app.live_vitals.tests import synth


def test_a_still_capture_is_read_whole(captured, frame_count):
    clip, fps, quality = captured("still_avi")
    assert quality.verdict == "ACCEPT"
    assert quality.n_frames == len(clip) == frame_count("still_avi")
    assert fps == pytest.approx(30.0, abs=0.01)
    assert quality.detections == quality.detections_attempted == config.N_DETECT_FRAMES
    assert quality.notes == [] and quality.reasons == []
    assert clip.shape[1:] == (config.TARGET_SIZE, config.TARGET_SIZE, 3)
    assert clip.dtype == np.uint8


def test_mp4_is_read_whole(captured, frame_count):
    clip, fps, quality = captured("still_mp4")
    assert quality.verdict == "ACCEPT"
    assert quality.n_frames == frame_count("still_mp4")
    assert fps == pytest.approx(30.0, abs=0.01)


@pytest.mark.parametrize("name", ["browser_vp8", "browser_vp9"])
def test_browser_webm_is_read_whole_at_its_real_rate(captured, frame_count, name):
    """MediaRecorder WebM declares no rate: OpenCV reports its 1 ms timebase as 1000 fps
    and derives a meaningless frame count, and a seek into the file goes astray. Every
    frame must still be read, at the rate its timestamps give."""
    clip, fps, quality = captured(name)
    assert quality.source_fps == pytest.approx(1000.0)
    assert quality.n_frames == frame_count(name)
    assert 29.0 < fps < 31.0
    assert quality.verdict == "ACCEPT"
    assert quality.detections == config.N_DETECT_FRAMES
    # A rate the container never declared is not reported as drift.
    assert not any("declares" in note for note in quality.notes)


@pytest.mark.parametrize("name, stride", [("still_avi", 1), ("browser_vp8", 1), ("fps60_avi", 2)])
def test_the_clip_is_every_strideth_frame_of_a_plain_decode(video, landmarker, monkeypatch,
                                                             name, stride):
    """No seek, no skip, no reordering: the crops equal those of a front-to-back decode."""
    seen = {}
    real_stable_box = session.stable_box

    def recording_stable_box(frames, lm):
        box, detections = real_stable_box(frames, lm)
        seen["box"] = box
        return box, detections

    monkeypatch.setattr(session, "stable_box", recording_stable_box)
    clip, _, _ = session.crops_from_video(video(name), landmarker)
    frames, _ = synth.sequential_frames(video(name))
    expected = np.asarray([crop_resize(f, seen["box"]) for f in frames[::stride]])
    assert clip.shape == expected.shape
    assert np.array_equal(clip, expected)


def test_60_fps_is_decimated_to_30(captured, frame_count):
    clip, fps, quality = captured("fps60_avi")
    assert quality.source_fps == pytest.approx(60.0)
    assert quality.n_frames == frame_count("fps60_avi") // 2
    assert fps == pytest.approx(30.0, abs=0.01)
    assert quality.verdict == "ACCEPT"


def test_a_low_rate_is_flagged_with_its_reason(captured):
    clip, fps, quality = captured("fps24_avi")
    assert fps == pytest.approx(24.0, abs=0.01)
    assert quality.verdict == "WARN"
    assert quality.reasons == ["24.0 fps is below the 30 fps training rate; treat as indicative"]


def test_a_small_head_movement_is_accepted(captured):
    _, _, quality = captured("small_motion_avi")
    assert quality.verdict == "ACCEPT" and quality.reasons == []
    assert 0.0 < quality.motion_px < config.MOTION_PERIODIC_PX


def test_a_rhythmic_head_movement_is_refused_as_exercise(captured):
    _, _, quality = captured("rhythm_avi")
    assert quality.verdict == "REJECT"
    assert len(quality.reasons) == 1 and "rhythmically" in quality.reasons[0]
    assert config.MOTION_PERIODIC_PX <= quality.motion_px < config.MOTION_REJECT_PX


def test_a_large_head_movement_is_refused(captured):
    _, _, quality = captured("bob_avi")
    assert quality.verdict == "REJECT" and not quality.usable
    assert len(quality.reasons) == 1 and quality.reasons[0].startswith("the head moved")
    assert quality.motion_px >= config.MOTION_REJECT_PX


def test_the_reason_is_the_refusing_gate_not_the_first_note(captured):
    """A portrait capture with a moving head carries the landscape warning first and the
    motion refusal second; only the refusal is the reason."""
    _, _, quality = captured("portrait_bob_avi")
    assert quality.verdict == "REJECT"
    assert quality.notes[0].endswith("record in landscape")
    assert quality.reasons == [quality.notes[-1]]
    assert quality.reasons[0].startswith("the head moved")


def test_no_face_is_an_error(video, landmarker):
    with pytest.raises(RuntimeError, match="no face detected"):
        session.crops_from_video(video("noface_avi"), landmarker)


def test_a_missing_file_is_an_error(tmp_path, landmarker):
    with pytest.raises(FileNotFoundError):
        session.crops_from_video(tmp_path / "absent.mp4", landmarker)


def test_every_video_opened_is_released_when_a_read_fails(video, landmarker, monkeypatch):
    """A decode that raises must not leave the file open: on Windows the web app could
    not then delete its copy of the upload."""
    real_capture = cv2.VideoCapture
    counts = dict(opened=0, released=0)

    class Counted:
        def __init__(self, *args):
            self._cap = real_capture(*args)
            counts["opened"] += 1

        def release(self):
            counts["released"] += 1
            self._cap.release()

        def __getattr__(self, name):
            return getattr(self._cap, name)

    calls = itertools.count()
    real_crop = session.crop_resize

    def failing(frame, box):
        if next(calls) == 20:
            raise RuntimeError("read failed")
        return real_crop(frame, box)

    monkeypatch.setattr(session.cv2, "VideoCapture", Counted)
    monkeypatch.setattr(session, "crop_resize", failing)
    with pytest.raises(RuntimeError, match="read failed"):
        session.crops_from_video(video("still_avi"), landmarker)
    assert counts["opened"] >= 3 and counts["released"] == counts["opened"]


def test_a_video_longer_than_max_frames_is_refused_before_cropping(video, landmarker,
                                                                    frame_count, monkeypatch):
    n = frame_count("short_avi")
    crops = itertools.count()
    real_crop = session.crop_resize

    def counting(frame, box):
        next(crops)
        return real_crop(frame, box)

    monkeypatch.setattr(session, "crop_resize", counting)
    with pytest.raises(ValueError, match=f"has more than {n - 1} frames"):
        session.crops_from_video(video("short_avi"), landmarker, max_frames=n - 1)
    assert next(crops) == 0
    clip, _, _ = session.crops_from_video(video("short_avi"), landmarker, max_frames=n)
    assert len(clip) == n