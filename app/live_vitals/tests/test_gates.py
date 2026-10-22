"""Capture gates in isolation: frame rate, decimation, framing, head motion, verdicts."""
import cv2
import numpy as np
import pytest

from app.live_vitals import config
from app.live_vitals.capture import session
from app.live_vitals.capture.motion import motion_verdict
from app.live_vitals.preprocess.face_box import (
    box_clamp_report, share_inside, square_padded_box)
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

def stores_side_lost(box):
    """box_side_lost as NB_P3_27 and NB_P3_29 record it in every Phase-3 store."""
    lost_x = max(0.0, -box[0]) + max(0.0, box[2] - 1.0)
    lost_y = max(0.0, -box[1]) + max(0.0, box[3] - 1.0)
    return max(lost_x / (box[2] - box[0]), lost_y / (box[3] - box[1]))


@pytest.mark.parametrize("width, height", [(640, 480), (1280, 720), (480, 640)])
def test_overflow_is_measured_against_its_own_side(width, height):
    # 0.03 of a 0.4-tall box lies above the frame: 7.5% of its height, whatever the frame.
    report = box_clamp_report((0.3, -0.03, 0.7, 0.37), width, height)
    assert report["frac_side_lost"] == pytest.approx(0.075)


def test_a_face_cut_off_at_the_top_of_a_wide_frame_is_refused():
    # 3% of the box's height lies above a 16:9 frame. Measured against the box's width
    # it would read 1.7%, inside the limit.
    verdict, report, notes = session.framing_verdict((0.3, -0.012, 0.7, 0.388), 1280, 720)
    assert report["frac_side_lost"] == pytest.approx(0.03)
    assert verdict == "REJECT"
    assert notes == ["face box extends past the frame; sit further back"]


def test_the_worse_axis_decides_not_the_sum():
    # 1.5% of the box lost on each axis: neither is past the limit.
    verdict, report, _ = session.framing_verdict((-0.006, -0.006, 0.394, 0.394), 640, 480)
    assert report["frac_side_lost"] == pytest.approx(0.015)
    assert verdict == "ACCEPT"


def test_the_app_measures_clamping_as_the_stores_do():
    rng = np.random.default_rng(0)
    for _ in range(200):
        cx, cy, half = rng.uniform(-0.1, 1.1), rng.uniform(-0.1, 1.1), rng.uniform(0.05, 0.4)
        box = (cx - half, cy - half, cx + half, cy + half)
        for width, height in ((640, 480), (1920, 1080), (960, 1056)):
            assert (box_clamp_report(box, width, height)["frac_side_lost"]
                    == pytest.approx(stores_side_lost(box)))

            
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
    assert quality.second_face_samples == 0 and np.isnan(quality.face_inside_min)
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


# ------------------------------------------------------------- face checks --

# A subject's landmark box, and the box a capture would hold for it: 1.6 times its longer
# side, square, around its centre - (0.18, 0.18, 0.82, 0.82).
SUBJECT = (0.40, 0.30, 0.60, 0.70)
HELD = square_padded_box([SUBJECT])


def moved(box, dx):
    return (box[0] + dx, box[1], box[2] + dx, box[3])


def test_share_inside():
    assert share_inside(SUBJECT, HELD) == pytest.approx(1.0)
    # 0.6 + 0.26 reaches 0.04 past the held box's right edge at 0.82: a fifth of the width.
    assert share_inside(moved(SUBJECT, 0.26), HELD) == pytest.approx(0.8)
    assert share_inside(moved(SUBJECT, 0.70), HELD) == 0.0


@pytest.mark.parametrize("found, verdict, note", [
    (12, "ACCEPT", None),
    (11, "WARN", "the face was missed in 1 of 12 sample frames; treat as indicative"),
    (10, "WARN", "the face was missed in 2 of 12 sample frames; treat as indicative"),
    (9, "REJECT", "the face was found in only 9 of 12 sample frames; keep your whole face in view"),
    (3, "REJECT", "the face was found in only 3 of 12 sample frames; keep your whole face in view")])
def test_faces_found_in_the_samples(found, verdict, note):
    faces = [[SUBJECT]] * found + [[]] * (12 - found)
    got, gate_notes, summary = session.face_verdict(faces, HELD)
    assert got == verdict
    assert gate_notes == ([] if note is None else [(verdict, note)])
    assert summary == dict(second_face_samples=0, face_inside_min=pytest.approx(1.0))


@pytest.mark.parametrize("ratio, counted", [(0.9, True), (0.55, True), (0.45, False)])
def test_a_second_face_warns_from_half_the_subjects_width(ratio, counted):
    width = (SUBJECT[2] - SUBJECT[0]) * ratio
    other = (0.70, 0.40, 0.70 + width, 0.40 + 2 * width)
    faces = [[SUBJECT, other]] * 2 + [[SUBJECT]] * 10
    verdict, gate_notes, summary = session.face_verdict(faces, HELD)
    assert summary["second_face_samples"] == (2 if counted else 0)
    assert verdict == ("WARN" if counted else "ACCEPT")
    if counted:
        assert gate_notes == [("WARN", "a second face was in view in 2 of 12 sample frames; "
                                       "only one person should face the camera")]


@pytest.mark.parametrize("dx, verdict, note", [
    (0.22, "ACCEPT", None),  # at the held box's edge, wholly inside
    (0.26, "WARN", "part of the face left the analysed area in 1 of 12 sample frames; "
                   "treat as indicative"),
    (0.36, "REJECT", "the face moved out of the analysed area (at worst 30% of it inside); "
                     "keep still and centred")])
def test_a_face_outside_the_held_box(dx, verdict, note):
    faces = [[SUBJECT]] * 11 + [[moved(SUBJECT, dx)]]
    got, gate_notes, summary = session.face_verdict(faces, HELD)
    assert got == verdict
    assert gate_notes == ([] if note is None else [(verdict, note)])
    assert summary["face_inside_min"] == pytest.approx(1.0 - max(0.0, 0.6 + dx - 0.82) / 0.2)


def test_every_face_check_reports_at_its_own_level():
    """A missed sample, a second face and a face far outside the box at once."""
    other = (0.70, 0.40, 0.85, 0.70)
    faces = [[]] + [[SUBJECT, other]] + [[SUBJECT]] * 9 + [[moved(SUBJECT, 0.36)]]
    verdict, gate_notes, _ = session.face_verdict(faces, HELD)
    assert verdict == "REJECT"
    assert [level for level, _ in gate_notes] == ["WARN", "WARN", "REJECT"]