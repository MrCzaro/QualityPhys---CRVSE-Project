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

