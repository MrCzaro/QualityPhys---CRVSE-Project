# app/live_vitals/config.py
"""Single source of truth for the Phase-3 train/serve contract.

Every constant here is fixed by how the Phase-3 models were trained
(NB_P3_03/05/06 preprocessing, NB_P3_07/18/22 loaders). Changing any value
silently moves inference off-distribution.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#  preprocessing contract (NB_P3_05) 
TARGET_SIZE = 72 # INTER_AREA downsample, RGB uint8
CROP_PADDING = 0.6 # square box = max(w,h) * (1 + padding)
N_DETECT_FRAMES = 12 # frames sampled for the median stable box
MP_MODEL = REPO_ROOT / "app/live_hr_demo/models/mediapipe/face_landmarker.task"

#  model input contract (NB_P3_07 / NB_P3_18) 
CLIP_LEN = 160 # frames per clip
TARGET_FPS = 30.0 # training corpus rate; higher-rate input is decimated
HR_LOW_HZ, HR_HIGH_HZ = 0.66, 3.0

#  checkpoints 
CKPT_DIR = REPO_ROOT / "Data/Phase3"
PHYSNET_CKPT = CKPT_DIR / "phase_3_physnet_v2_baseline_best.zip"
RHYTHMFORMER_CKPT = CKPT_DIR / "phase_3_rhythmformer_best.zip"

#  capture-quality gates (app-side; not part of the training contract) 
MAX_BOX_SIDE_LOST = 0.02  # reject if the padded box is clamped by more than this

# Known training video is 4:3 (UBFC and MCD are both 640x480). Landscape 16:9 was
# validated empirically at 56.80 bpm against an oximeter reading 56-58, so it is
# accepted too. A portrait frame inverts the stretch and cost +2.4 bpm on an
# otherwise identical resting capture, so it is flagged even when nothing is clamped.
TRAIN_ASPECT_RANGE = (1.20, 1.90)

# HR readout (app-side; not part of the training contract) 
FFT_PAD_FACTOR = 8 # zero-pad before the FFT to escape coarse bin spacing
SUBHARMONIC_RATIO = 0.20  # adopt f/2 when its power exceeds this fraction of the peak
# Disabled by default: on real rPPG the guard produced a false halving at high HR
# (subject11 w19, 124 -> 64 bpm) and no true corrections, costing 2.5 bpm of MAE.
HARMONIC_GUARD_ENABLED = False
WINDOW_STRIDE = CLIP_LEN // 2

# Windows below this spectral concentration are discarded, not just flagged.
# Confidence scales with how much of the crop the face fills, which is set by frame
# aspect, so an absolute threshold does not transfer between framings: 0.80 kept
# 16/21 windows on a portrait clip but only 4/21 on an equally clean 16:9 one.
# 0.65 transfers across portrait, 16:9 and motion captures (measured 2026-08-29).
MIN_CONFIDENCE = 0.65
MAD_OUTLIER_K = 3.0   # reject windows this many MADs from the median HR
MAD_FLOOR_BPM = 1.0   # never reject inside this band, however tight the cluster
MIN_USABLE_WINDOWS = 3    # fewer surviving windows than this is not a reading
MIN_USABLE_FRACTION = 0.50  # a low surviving fraction is itself a motion signal
# IQR spread tracks genuine within-capture HR variation more than error (r~0.48 against
# MAE on the 8 held-out subjects), so 10 produced false alarms on three good captures
# including the second-most-accurate. Kept only as a wide backstop; usable_fraction is
# the reliable instability signal.
MAX_HR_SPREAD_BPM = 25.0

# Confidence lobe half-width floor. The lobe is otherwise fps/N Hz -- the Rayleigh
# resolution -- which shrinks as the window lengthens while the spectral peak does
# not, because a real heart rate wanders within a long window. Concentration then
# falls purely because the window grew, and MIN_CONFIDENCE silently means something
# different at every window length: measured on the MCD subset, POS median
# confidence fell 0.630 -> 0.469 and the kept fraction 0.51 -> 0.39 going from 5.3 s
# to 30 s windows. Flooring the lobe at a bandwidth the heart rate genuinely occupies
# holds the kept fraction at 0.51 across all five lengths. 0.15 Hz is about 9 bpm,
# and sits below fps/N for any capture the rate gate accepts at CLIP_LEN, so it does
# not engage at the production window: per-window confidence over the eight held-out
# UBFC subjects is unchanged to the last bit.
CONFIDENCE_FLOOR_HZ = 0.15

# Acquisition rate gates. The 2026-07-21 sweep found good tracking needs 28-30 Hz
# and that 22-26 Hz is an unreliable dead zone; a 10 fps webcam capture produced a
# confident reading ~7 bpm below a known resting HR. Rate is a hard contract, not
# a note: confidence cannot detect a wrong time base.
MIN_FPS_ACCEPT = 27.0   # at or above this, rate is not a concern
MIN_FPS_WARN = 20.0     # below this the reading is refused outright

# Below this surviving fraction the median is dominated by noise, so no value is
# reported at all. A qualified wrong number is worse than a refusal: a capture that
# loses 86% of its windows has not measured anything.
MIN_REPORTABLE_FRACTION = 0.25


# Classical spectral cross-check (app-side; not part of the training contract)
SPECTRAL_METHOD = "pos" # reported method; "chrom" and "green" also available
SPECTRAL_WINDOW_SECONDS = 1.6 # projection window used by both POS and CHROM
SPECTRAL_FILTER_ORDER = 4 # Butterworth order for the cardiac bandpass
# Fraction of the face crop, measured from its centre, averaged into the RGB
# trace. CROP_PADDING leaves the outer crop full of hair, shoulders and
# background whose brightness changes are not pulsatile. Swept on the eight
# held-out UBFC subjects (2026-09-01): CHROM window MAE 2.68 bpm over the whole
# crop, 1.38 at 0.45, 1.29 at 0.35, 1.26 at 0.30, then 1.44 at 0.25 as the
# region starts to miss skin. 0.35 sits on the flat part of that curve, so it
# tolerates a face that is not perfectly centred in its box.
SPECTRAL_ROI_FRACTION = 0.35
# POS is the default rather than CHROM. CHROM wins narrowly on UBFC (1.29 bpm
# window MAE against 1.35) but loses badly on the six held-out MCD-rPPG subjects,
# where it is refused on 9 of 12 recordings to POS's 5 and carries 24.9 bpm window
# RMSE against 18.4. POS is also the classical method Egorov et al. (ACMMM 2025)
# benchmark on this corpus, at 3.80 bpm MAE on the frontal camera.