"""Capture gates in isolation: frame rate, decimation, framing, head motion, verdicts."""
import cv2
import numpy as np
import pytest

from app.live_vitals import config
from app.live_vitals.capture import session
from app.live_vitals.capture.motion import motion_verdict
from app.live_vitals.tests import synth

NAN = float("nan")


# ------------------------------------------------------------- frame rate --

def test_measured_rate_comes_from_the_timestamps():
    stamps = [1000.0 * i / 29.5 for i in range(300)]
    fps, note = session.measured_rate(stamps, 30.0)
    assert fps == pytest.approx(29.5) and note is None


@pytest.mark.parametrize("stamps", [[], [0.0], [5.0, 5.0, 5.0]])
def test_without_usable_timestamps_the_declared_rate_is_used_and_said_so(stamps):
    fps, note = session.measured_rate(stamps, 30.0)
    assert fps == 30.0 and "timestamps unavailable" in note


@pytest.mark.parametrize("rate, stride", [(30.0, 1), (29.97, 1), (24.0, 1), (44.0, 1),
                                          (60.0, 2), (59.94, 2), (90.0, 3),
                                          (0.0, 1), (-5.0, 1), (NAN, 1)])
def test_decimation_stride_brings_the_rate_near_30(rate, stride):
    assert session.decimation_stride(rate) == stride


@pytest.mark.parametrize("fps, verdict", [
    (NAN, "REJECT"), (10.0, "REJECT"), (config.MIN_FPS_WARN - 0.1, "REJECT"),
    (config.MIN_FPS_WARN, "WARN"), (config.MIN_FPS_ACCEPT - 0.1, "WARN"),
    (config.MIN_FPS_ACCEPT, "ACCEPT"), (30.0, "ACCEPT"), (config.MAX_FPS_ACCEPT, "ACCEPT"),
    (config.MAX_FPS_ACCEPT + 0.1, "WARN"), (60.0, "WARN")])
def test_rate_verdict(fps, verdict):
    got, note = session.rate_verdict(fps)
    assert got == verdict
    assert (note is None) == (verdict == "ACCEPT")


# ---------------------------------------------------------------- framing --

def test_a_centred_box_on_a_landscape_frame_is_accepted():
    verdict, report, notes = session.framing_verdict((0.4, 0.4, 0.6, 0.6), 640, 480)
    assert verdict == "ACCEPT" and notes == []
    assert report["expected_aspect"] == pytest.approx(640 / 480)


def test_a_box_past_the_frame_is_refused():
    verdict, report, notes = session.framing_verdict((-0.1, 0.4, 0.1, 0.6), 640, 480)
    assert verdict == "REJECT"
    assert report["frac_side_lost"] > config.MAX_BOX_SIDE_LOST
    assert notes == ["face box extends past the frame; sit further back"]


def test_a_portrait_frame_is_flagged():
    verdict, report, notes = session.framing_verdict((0.4, 0.4, 0.6, 0.6), 480, 640)
    assert verdict == "WARN" and "record in landscape" in notes[0]


# --------------------------------------------------------------- verdicts --

def test_the_worst_verdict_wins():
    assert session.worst_verdict("ACCEPT", "ACCEPT") == "ACCEPT"
    assert session.worst_verdict("ACCEPT", "WARN") == "WARN"
    assert session.worst_verdict("WARN", "REJECT", "ACCEPT") == "REJECT"


def test_capture_quality_defaults():
    quality = session.CaptureQuality(640, 480, 30.0, 30.0, 600, 12, 12, "ACCEPT", 1.333, 0.0)
    assert quality.notes == [] and quality.reasons == []
    assert np.isnan(quality.motion_px) and quality.usable
    refused = session.CaptureQuality(640, 480, 30.0, 30.0, 600, 12, 12, "REJECT", 1.333, 0.0)
    assert not refused.usable


# ------------------------------------------------------ detection samples --

def test_detection_samples_span_the_whole_recording():
    positions = session.detection_positions(600)
    assert len(positions) == config.N_DETECT_FRAMES
    assert positions[0] == 0 and positions[-1] == 599
    assert positions == sorted(set(positions))


@pytest.mark.parametrize("n", [0, -5, -2 ** 62])
def test_a_meaningless_count_samples_only_the_first_frame(n):
    # OpenCV reports a negative count for a streamed WebM; the second pass then re-places them.
    assert session.detection_positions(n) == [0]


def test_fewer_frames_than_samples():
    assert session.detection_positions(5) == [0, 1, 2, 3, 4]


# ------------------------------------------------------------ head motion --

def shifting_crops(amplitude_px, hz=1.2, n=360, fps=30.0):
    """72x72 face crops whose content moves vertically by amplitude_px at hz."""
    face = synth.portrait()
    h, w = face.shape[:2]
    patch = cv2.resize(face[int(0.15 * h):int(0.85 * h), int(0.1 * w):int(0.9 * w)],
                       (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    crops = []
    for i in range(n):
        shift = np.float32([[1, 0, 0], [0, 1, amplitude_px * np.sin(2 * np.pi * hz * i / fps)]])
        moved = cv2.warpAffine(patch, shift, (96, 96), borderMode=cv2.BORDER_REFLECT)
        crops.append(cv2.resize(moved[12:84, 12:84], (72, 72), interpolation=cv2.INTER_AREA))
    return np.clip(np.rint(crops), 0, 255).astype(np.uint8)


def test_a_still_head_is_accepted():
    verdict, note, summary = motion_verdict(shifting_crops(0.0), 30.0)
    assert verdict == "ACCEPT" and note is None
    assert summary["amp_median"] < 0.05


def test_a_small_rhythm_is_accepted():
    # A rhythm no larger than a resting head's own never refuses a capture.
    verdict, note, summary = motion_verdict(shifting_crops(0.5), 30.0)
    assert verdict == "ACCEPT"
    assert summary["clear_frac"] >= config.MOTION_PERIODIC_SHARE
    assert summary["amp_median"] < config.MOTION_PERIODIC_PX


def test_a_moderate_rhythm_is_refused_as_exercise():
    verdict, note, summary = motion_verdict(shifting_crops(2.0), 30.0)
    assert verdict == "REJECT" and "rhythmically" in note
    assert config.MOTION_PERIODIC_PX <= summary["amp_median"] < config.MOTION_REJECT_PX


def test_a_large_movement_is_refused():
    verdict, note, summary = motion_verdict(shifting_crops(4.0), 30.0)
    assert verdict == "REJECT" and note.startswith("the head moved")
    assert summary["amp_median"] >= config.MOTION_REJECT_PX


@pytest.mark.parametrize("n, fps", [(100, 30.0), (360, NAN), (360, 0.0)])
def test_motion_is_not_judged_without_a_window_or_a_rate(n, fps):
    assert motion_verdict(shifting_crops(4.0, n=n), fps) == ("ACCEPT", None, None)