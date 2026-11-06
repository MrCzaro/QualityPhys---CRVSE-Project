"""Respiratory rate from the torso's motion, on synthetic signals and frames with known answers:
the readout (`signal.respiration`), then the development check's guards and scoring
(`scripts.check_rr`). Nothing here reads a store."""
import uuid

import h5py
import numpy as np
import pandas as pd
import pytest

from app.live_vitals.scripts import check_rr
from app.live_vitals.signal.respiration import (FULL_FS, MIN_FRAMES, MIN_TORSO_ROWS, N_COMPONENTS,
                                                TorsoReading, breathing_band, components,
                                                peak_share, resp_rate, torso_rate, torso_region)

SECONDS = 60
BOX_RY1 = 0.35  # the torso region then holds the lower 85 of 132 rows


def tone(bpm, n=int(SECONDS * FULL_FS), fs=FULL_FS, phase=0.3):
    """A sinusoid at bpm breaths/min, sampled at fs."""
    return np.sin(2 * np.pi * bpm / 60.0 * np.arange(n) / fs + phase)


def torso_frames(bpm, fs=FULL_FS, amplitude_px=0.3, noise=2.0, seed=0, height=132, width=120):
    """A whole frame at frames_full's geometry whose lower part - a horizontal edge (the
    shoulders against the background) and a smooth texture - rises and falls by amplitude_px
    at bpm, under pixel noise and a slow brightening, quantised to uint8."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(SECONDS * fs))) / fs
    dy = amplitude_px * np.sin(2 * np.pi * bpm / 60.0 * t + 0.4)
    y = np.arange(height)[None, :, None] - dy[:, None, None]
    x = np.arange(width)[None, None, :]
    shoulder = 1.0 / (1.0 + np.exp(-(y - 0.62 * height) / 2.5))
    texture = (np.sin(2 * np.pi * y / 17.0 + 0.7) * np.cos(2 * np.pi * x / 29.0)
               + 0.5 * np.sin(2 * np.pi * (y / 9.0 + x / 41.0)))
    image = 90.0 + 70.0 * shoulder + 12.0 * texture + 4.0 * t[:, None, None] / SECONDS
    frames = image[..., None] * np.array([1.0, 0.85, 0.7]) + rng.normal(0.0, noise, image.shape + (3,))
    return np.clip(np.rint(frames), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- the readout --

@pytest.mark.parametrize("bpm", [7.3, 13.7, 22.1, 37.9])
def test_resp_rate_reads_a_tone_between_spectral_bins(bpm):
    assert resp_rate(tone(bpm), FULL_FS) == pytest.approx(bpm, abs=0.01)


@pytest.mark.parametrize("bpm", [5.5, 42.5])
def test_resp_rate_refuses_a_peak_on_either_band_edge(bpm):
    # Just outside 6-42 breaths/min, the band's largest power sits on its edge.
    assert np.isnan(resp_rate(tone(bpm), FULL_FS))


def test_resp_rate_refuses_what_it_cannot_read():
    assert np.isnan(resp_rate(np.ones(600), FULL_FS))  # flat
    assert np.isnan(resp_rate(tone(15.0, n=15), FULL_FS))  # too short
    assert np.isnan(resp_rate(np.r_[tone(15.0)[:-1], np.nan], FULL_FS))  # not finite


def test_peak_share_is_high_for_a_tone_and_low_for_noise():
    assert peak_share(tone(15.0), FULL_FS) > 0.95
    rng = np.random.default_rng(1)
    shares = [peak_share(rng.normal(size=600), FULL_FS) for _ in range(5)]
    assert all(0.0 < s < 0.5 for s in shares)


def test_torso_region_is_grey_rows_below_the_face_box():
    frames = np.zeros((3, 132, 120, 3), np.uint8)
    frames[..., 0], frames[..., 1], frames[..., 2] = 30, 60, 90
    region = torso_region(frames, 0.5)
    assert region.shape == (3, 66, 120)  # rows from ceil(0.5 x 132) to the bottom
    assert np.all(region == 60.0)


def test_torso_region_needs_eight_rows():
    frames = np.zeros((3, 132, 120, 3), np.uint8)
    assert torso_region(frames, 123.5 / 132).shape[1] == MIN_TORSO_ROWS
    assert torso_region(frames, 124.5 / 132) is None


def test_components_equal_the_svd_up_to_sign():
    rng = np.random.default_rng(3)
    region = rng.normal(size=(120, 9, 11))
    found = components(region, FULL_FS)
    X = breathing_band(region.reshape(120, -1), FULL_FS)
    U, S, _ = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    assert found.shape == (N_COMPONENTS, 120)
    for k in range(N_COMPONENTS):
        expected = U[:, k] * S[k]
        sign = np.sign(found[k] @ expected)
        np.testing.assert_allclose(found[k], sign * expected, atol=1e-9 * np.linalg.norm(expected))


@pytest.mark.parametrize("bpm", [12.4, 18.7, 29.3])
def test_torso_rate_reads_a_breathing_torso(bpm):
    reading = torso_rate(torso_frames(bpm), BOX_RY1)
    assert reading.rate_bpm == pytest.approx(bpm, abs=0.1)
    assert 0.0 < reading.share <= 1.0
    assert 0 <= reading.component < N_COMPONENTS
    assert reading.rows == 85


def test_torso_rate_follows_the_frame_rate_it_is_given():
    # Only 10 fps has been tested on real video; this checks nothing assumes it.
    reading = torso_rate(torso_frames(18.7, fs=15.0), BOX_RY1, fs=15.0)
    assert reading.rate_bpm == pytest.approx(18.7, abs=0.1)


def test_torso_rate_refuses_without_torso_rows():
    reading = torso_rate(torso_frames(18.7)[:20], 0.97)
    assert isinstance(reading, TorsoReading)
    assert np.isnan(reading.rate_bpm) and np.isnan(reading.share)
    assert (reading.component, reading.rows) == (-1, 0)


def test_torso_rate_refuses_a_still_torso_and_too_few_frames():
    still = torso_rate(np.full((600, 132, 120, 3), 120, np.uint8), BOX_RY1)
    short = torso_rate(torso_frames(18.7)[:MIN_FRAMES - 1], BOX_RY1)
    for reading in (still, short):
        assert np.isnan(reading.rate_bpm) and np.isnan(reading.share)
        assert (reading.component, reading.rows) == (-1, 85)


def test_torso_rate_rejects_malformed_input():
    frames = torso_frames(18.7)[:40]
    with pytest.raises(ValueError):
        torso_rate(frames[..., 0], BOX_RY1)  # no colour axis
    with pytest.raises(ValueError):
        torso_rate(frames[..., :2], BOX_RY1)  # two channels
    for box in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            torso_rate(frames, box)
    with pytest.raises(ValueError):
        torso_rate(frames, BOX_RY1, fs=2.0)  # the band-pass needs more than 2 Hz


# ------------------------------------------------------- the development check --

def _store(splits):
    """An in-memory store whose recordings carry only a split and a tiny frames_full."""
    store = h5py.File(f"{uuid.uuid4().hex}.h5", "w", driver="core", backing_store=False)
    for guid, split in splits.items():
        group = store.create_group(guid)
        group.attrs["split"] = split
        group.create_dataset("frames_full", data=np.zeros((2, 4, 4, 3), np.uint8))
    return store


def test_frame_reader_refuses_held_out_recordings():
    with _store({"a1": "train", "b2": "val"}) as store:
        assert check_rr.read_frames(store["a1"], "frames_full").shape == (2, 4, 4, 3)
        with pytest.raises(RuntimeError, match="held-out"):
            check_rr.read_frames(store["b2"], "frames_full")


def test_training_guids_require_the_frozen_split(tmp_path):
    frozen = tmp_path / "split.csv"
    pd.DataFrame(dict(guid=["b2", "a1", "c3"], split=["val", "train", "train"])).to_csv(frozen, index=False)
    with _store({"a1": "train", "b2": "val", "c3": "train"}) as store:
        assert check_rr.train_guids(store, frozen) == ["a1", "c3"]
    with _store({"a1": "train", "b2": "train", "c3": "train"}) as store:
        with pytest.raises(SystemExit):
            check_rr.train_guids(store, frozen)


def test_vertical_shift_follows_a_sub_pixel_shift():
    from scipy.ndimage import fourier_shift, gaussian_filter
    rng = np.random.default_rng(0)
    base = gaussian_filter(rng.normal(size=(60, 80)), 2.0) * 40 + 128
    truth = 0.8 * np.sin(2 * np.pi * np.arange(200) / 40.0)
    region = np.stack([np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(base), (s, 0)))) for s in truth])
    found = check_rr.vertical_shift(region)
    found, truth = found - found.mean(), truth - truth.mean()
    assert np.corrcoef(found, truth)[0, 1] > 0.999
    assert np.sqrt(np.mean((found - truth) ** 2)) < 0.05


def test_wilson_interval_matches_the_development_figure():
    lo, hi = check_rr.wilson(197, 208)  # NB_P3_39: 95% 90.8%-97.0%
    assert (round(lo, 3), round(hi, 3)) == (0.908, 0.970)
    assert all(np.isnan(v) for v in check_rr.wilson(0, 0))


def test_gate_counts_an_unread_recording_as_a_miss():
    scored = pd.DataFrame(dict(belt=[15.0, 20.0, 20.0, 10.0, 21.0],
                               a_rate=[15.5, np.nan, 20.5, 13.0, 25.0]))
    g = check_rr.gate(scored, "a_rate", baseline=20.0)
    # Read 4 of 5; right: the first and third; the baseline (20) is right for the 2nd, 3rd, 5th.
    assert (g["read"], g["within_2"], g["within_1"]) == (4, 2, 2)
    assert g["coverage"] == pytest.approx(0.8) and g["share"] == pytest.approx(0.5)
    # Only the readout right: the first. Only the baseline right: the unread second and the fifth.
    assert (g["readout_only"], g["baseline_only"]) == (1, 2)
    assert g["outcome"] == "development negative"


def test_module_hash_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"A = 1\nB = 2\n")
    crlf.write_bytes(b"A = 1\r\nB = 2\r\n")
    assert check_rr.text_sha256(lf) == check_rr.text_sha256(crlf)
    crlf.write_bytes(b"A = 1\r\nB = 3\r\n")
    assert check_rr.text_sha256(lf) != check_rr.text_sha256(crlf)


def test_readings_digest_ignores_row_order_and_sees_the_sixth_decimal():
    frame = pd.DataFrame(dict(guid=["c3", "a1", "b2"], torso_rows=[40, 39, 41],
                              a_rate=[12.3456781, 18.2, 20.0], a_share=[0.5, 0.6, 0.7],
                              a_component=[0, 1, 0], b_rate=[12.1, np.nan, 19.8],
                              b_share=[0.4, np.nan, 0.65], belt=[12.0, 18.0, 20.0]))
    digest = check_rr.readings_digest(frame)
    assert check_rr.readings_digest(frame.iloc[::-1]) == digest
    assert check_rr.readings_digest(frame.assign(belt=0.0)) == digest  # only the readings count
    nudged = frame.copy()
    nudged.loc[0, "a_rate"] += 1e-9
    assert check_rr.readings_digest(nudged) == digest
    nudged.loc[0, "a_rate"] += 1e-6
    assert check_rr.readings_digest(nudged) != digest