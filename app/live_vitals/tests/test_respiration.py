"""Respiratory rate from the torso's motion, on synthetic signals and frames with known answers:
the readout (`signal.respiration`), then the development check's guards and scoring
(`scripts.check_rr`), its held-out stage included. Nothing here reads a store or a recording."""
import hashlib
import json
import sys
import uuid

import cv2
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


# ----------------------------------------------------------- the held-out stage --

def _held_out_store(rate):
    """An in-memory store holding one VV-60 recording: a synthetic breathing torso and the
    attributes scoring reads."""
    store = _store({})
    group = store.create_group("v1")
    group.create_dataset("frames_full", data=torso_frames(rate))
    group.attrs.update(split="val", full_fps=10.0, box_ry1=BOX_RY1, resp_ok=True, note_resp_bad=False,
                       site_group="MOL/RING", fitzpatrick=2, age=40, rr_belt=rate, rr_surrogate=np.nan)
    return store


def _portrait_video(path, bpm, fps=30.0, seconds=40, width=270, height=480):
    """A portrait video in the iPhone's framing: the suite's face near the top and, below it, a
    textured torso that rises and falls 3 px at bpm. Written with OpenCV (MJPG)."""
    from app.live_vitals.tests import synth
    face = synth.portrait()
    face_h = 170
    face_w = int(face_h * face.shape[1] / face.shape[0])
    face = cv2.resize(face, (face_w, face_h), interpolation=cv2.INTER_AREA)
    y = np.arange(height, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[None, :]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    for i in range(int(fps * seconds)):
        moved = y - 3.0 * np.sin(2 * np.pi * bpm / 60.0 * i / fps)
        torso = (90 + 60 / (1 + np.exp(-(moved - 300) / 4))
                 + 15 * np.sin(2 * np.pi * moved / 19) * np.cos(2 * np.pi * x / 31))
        image = np.repeat(np.clip(torso, 0, 255)[..., None], 3, axis=-1)
        image[:250] = 96.0
        image[30:30 + face_h, (width - face_w) // 2:(width - face_w) // 2 + face_w] = face
        writer.write(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def test_held_out_settings_are_part_0s():
    assert check_rr.SETTINGS == {
        "bar: Wilson 95% lower bound at least": 0.70,
        "bar: coverage at least": 0.80,
        "bar: McNemar p below": 0.05,
        "Wilson z": 1.959964,
        "ladder: deciding rung": "R2",
        "ladder: no effect, every move within (breaths/min)": 1.0,
        "ladder: effect, a move beyond (breaths/min)": 2.0,
        "ladder: effect, least recordings lost or moved": 2,
        "ladder: downscale from half size, frames averaged": (4, 3),
        "self-test: mean absolute difference, pixel means removed (grey levels)": 1.0,
        "iPhone: downscale, frames averaged": (8, 6),
    }


def test_held_out_reader_refuses_training_recordings():
    with _store({"a1": "train", "b2": "val"}) as store:
        assert check_rr.read_held_out(store["b2"], "frames_full").shape == (2, 4, 4, 3)
        with pytest.raises(RuntimeError, match="VV-60 only"):
            check_rr.read_held_out(store["a1"], "frames_full")


def test_held_out_guids_require_the_frozen_split(tmp_path):
    frozen = tmp_path / "split.csv"
    pd.DataFrame(dict(guid=["b2", "a1", "c3"], split=["val", "train", "val"])).to_csv(frozen, index=False)
    with _store({"a1": "train", "b2": "val", "c3": "val"}) as store:
        assert check_rr.held_out_guids(store, frozen) == ["b2", "c3"]
    with _store({"a1": "val", "b2": "val", "c3": "val"}) as store:
        with pytest.raises(SystemExit):
            check_rr.held_out_guids(store, frozen)


def test_held_out_recording_is_read_by_both_readouts():
    with _held_out_store(18.7) as store:
        row = check_rr.read_recording(store, "v1", reader=check_rr.read_held_out)
        assert row["a_rate"] == pytest.approx(18.7, abs=0.1)
        assert row["b_rate"] == pytest.approx(18.7, abs=0.5)
        assert (row["torso_rows"], row["resp_ok"], row["belt"]) == (85, True, 18.7)
        with pytest.raises(RuntimeError):
            check_rr.read_recording(store, "v1")  # the development stage's reader refuses it


@pytest.mark.parametrize("right, read, belt, expected", [
    (46, 56, 10.0, "confirmed"),  # 82.1%, lower bound 70.2%
    (45, 56, 10.0, "inconclusive"),  # 80.4%, lower bound 68.2%
    (40, 56, 10.0, "inconclusive"),  # 71.4%
    (39, 56, 10.0, "not confirmed"),  # 69.6%
    (42, 44, 10.0, "inconclusive"),  # 95.5% of those read, but coverage 78.6%
    (46, 56, 20.0, "not confirmed"),  # the baseline (20) is right everywhere: A is not ahead
])
def test_held_out_outcome_follows_part_0s_bar(right, read, belt, expected):
    rate = [belt] * right + [belt + 5.0] * (read - right) + [np.nan] * (56 - read)
    g = check_rr.gate(pd.DataFrame(dict(belt=[belt] * 56, a_rate=rate)), "a_rate", baseline=20.0)
    assert check_rr.held_out_outcome(g) == expected


@pytest.mark.parametrize("rung, expected", [
    ({"a": 10.2, "b": 12.0, "c": 15.9}, "no effect"),
    ({"a": 10.2, "b": 12.0, "c": 16.6}, "inconclusive"),  # one move over 1, none over 2
    ({"a": 10.2, "b": np.nan, "c": 15.0}, "inconclusive"),  # one lost
    ({"a": np.nan, "b": np.nan, "c": 15.0}, "effect"),  # two lost
    ({"a": 13.0, "b": np.nan, "c": 15.0}, "effect"),  # one moved by 3 and one lost
])
def test_rung_outcome_follows_the_ladder_rule(rung, expected):
    assert check_rr.rung_outcome({"a": 10.0, "b": 12.0, "c": 15.0}, rung)["outcome"] == expected


def test_rung_outcome_with_nothing_read_at_r0_is_inconclusive():
    outcome = check_rr.rung_outcome({"a": np.nan}, {"a": 10.0})
    assert (outcome["read_at_r0"], outcome["outcome"]) == (0, "inconclusive")


def test_reduced_video_matches_reducing_its_source(tmp_path):
    import av
    yy, xx = np.mgrid[0:48, 0:64]
    source = [np.clip(np.stack([100 + 60 * np.sin(xx / 7 + i / 5), 120 + 50 * np.cos(yy / 5 - i / 7),
                                90 + 0 * xx + i], axis=-1), 0, 255).astype(np.uint8) for i in range(30)]
    path = tmp_path / "lossless.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv444p"
        stream.options = {"qp": "0", "preset": "veryfast"}
        for rgb in source:
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    frames, samples, n, fps = check_rr.reduced_video(path, 4, 3, keep=[0, 7])
    expected = np.stack([np.rint(np.mean([cv2.resize(f, (16, 12), interpolation=cv2.INTER_AREA)
                                          .astype(np.float32) for f in source[k:k + 3]], axis=0))
                         for k in range(0, 30, 3)])
    assert frames.shape == (10, 12, 16, 3) and frames.dtype == np.uint8
    assert (n, fps, sorted(samples)) == (30, 30.0, [0, 7])
    assert np.abs(frames.astype(int) - expected.astype(int)).max() <= 2  # the 4:4:4 colour round trip
    assert np.abs(samples[7].astype(int) - source[7].astype(int)).max() <= 3


def test_torso_difference_ignores_offsets_and_sees_change():
    stored = torso_frames(18.7)[:30]
    assert check_rr.torso_difference(stored, stored, BOX_RY1) == 0.0
    assert check_rr.torso_difference(stored + np.uint8(3), stored, BOX_RY1) < 1e-9  # gone with the means
    changed = stored.copy()
    changed[5] += np.uint8(6)  # one frame of 30 brighter by 6 levels
    assert check_rr.torso_difference(changed, stored, BOX_RY1) == pytest.approx(2 * 6 * 29 / 30 ** 2)
    assert np.isnan(check_rr.torso_difference(stored[:29], stored, BOX_RY1))  # another shape
    assert np.isnan(check_rr.torso_difference(stored, stored, 0.97))  # no torso region


def test_verify_ladder_holds_every_file_to_its_manifest(tmp_path):
    from app.live_vitals.scripts.bench_outputs import file_sha256
    names = ("g1/R0.mp4", "g1/R2.mp4")
    (tmp_path / "g1").mkdir()
    (tmp_path / names[0]).write_bytes(b"lossless")
    (tmp_path / names[1]).write_bytes(b"lossy!")
    manifest = dict(repository=dict(commit="abc", dirty=False, changed=[]), recordings=["g1"],
                    files={n: dict(bytes=(tmp_path / n).stat().st_size, sha256=file_sha256(tmp_path / n))
                           for n in names})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert check_rr.verify_ladder(tmp_path, commit="abc")[1] == []
    assert check_rr.verify_ladder(tmp_path, commit="abd")[1]  # written from another commit
    (tmp_path / names[1]).write_bytes(b"lossy?")  # the same size, other bytes
    assert any("SHA-256" in p for p in check_rr.verify_ladder(tmp_path, commit="abc")[1])
    (tmp_path / names[0]).write_bytes(b"lossles")
    assert any("bytes" in p for p in check_rr.verify_ladder(tmp_path, commit="abc")[1])


def test_files_fingerprint_is_the_freeze_records(tmp_path):
    (tmp_path / "g1").mkdir()
    (tmp_path / "g1" / "R0.mp4").write_bytes(b"12345")
    movie = tmp_path / "raw" / "g1" / "IMG_0001.MOV"
    movie.parent.mkdir(parents=True)
    movie.write_bytes(b"123")
    expected = hashlib.sha256(b"g1/R0.mp4 5\ng1/IMG_0001.MOV 3\n").hexdigest()
    assert check_rr.files_fingerprint(dict(files={"g1/R0.mp4": {}}), tmp_path, [movie]) == expected


def test_concentration_bands_cut_at_the_development_quintiles():
    scored = pd.DataFrame(dict(belt=[10.0] * 6, a_rate=[10.0, 10.0, 10.0, 15.0, 10.0, np.nan],
                               a_share=[0.1, 0.2, 0.45, 0.65, 0.9, np.nan]))
    bands = check_rr.concentration_bands(scored, [0.2, 0.4, 0.6, 0.8])
    assert [b["read"] for b in bands] == [1, 1, 1, 1, 1]  # a share on a cut joins the band above
    assert [b["right"] for b in bands] == [1, 1, 1, 0, 1]


def test_strata_count_right_over_read():
    scored = pd.DataFrame(dict(belt=[8.0, 15.0, 22.4, 30.0], a_rate=[8.5, 18.0, 22.0, np.nan],
                               site_group=["CH*", "CH*", "MOL/RING", "MOL/RING"],
                               fitzpatrick=[6, 5, 2, 1], age=[30, 45, 61, 70]))
    found = check_rr.strata(scored, "a_rate")
    assert found["Fitzpatrick"]["5-6"] == dict(n=2, read=2, right=1)
    assert found["age"]["41-60"] == dict(n=1, read=1, right=0)
    assert found["age"][">60"] == dict(n=2, read=1, right=1)
    assert found["belt rate"]["<=8"] == dict(n=1, read=1, right=1)
    assert found["belt rate"][">=25"] == dict(n=1, read=0, right=0)


def test_iphone_reading_finds_the_face_and_reads_the_torso_below(tmp_path, landmarker):
    row = check_rr.iphone_reading(_portrait_video(tmp_path / "portrait.avi", 15.0), landmarker)
    assert row["box_found"] and row["detections"] == 12
    # The padded box ends below the chin (the face spans 0.06-0.42 of the height), and the torso
    # region is every row of the 60-row reduced frame below that edge.
    assert 0.30 < row["box_ry1"] < 0.45
    assert row["rows"] == 60 - int(np.ceil(row["box_ry1"] * 60))
    assert row["rate"] == pytest.approx(15.0, abs=0.3)


def test_confirm_runs_whole_and_once(tmp_path, monkeypatch):
    # Should a guard ever fail, the stage must not start for real: it would read held-out frames.
    def started(args):
        pytest.fail("the guard let the held-out stage start")
    monkeypatch.setattr(check_rr, "run_confirm", started)
    monkeypatch.setattr(check_rr, "run_dev", started)
    monkeypatch.setattr(sys, "argv", ["check_rr", "--stage", "confirm", "--limit", "3",
                                      "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="runs whole"):
        check_rr.main()
    (tmp_path / "summary_confirm.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_rr", "--stage", "confirm", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="runs once"):
        check_rr.main()