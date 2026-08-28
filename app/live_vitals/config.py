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

# HR readout (app-side; not part of the training contract) 
FFT_PAD_FACTOR = 8 # zero-pad before the FFT to escape coarse bin spacing
SUBHARMONIC_RATIO = 0.20  # adopt f/2 when its power exceeds this fraction of the peak
# Disabled by default: on real rPPG the guard produced a false halving at high HR
# (subject11 w19, 124 -> 64 bpm) and no true corrections, costing 2.5 bpm of MAE.
HARMONIC_GUARD_ENABLED = False
WINDOW_STRIDE = CLIP_LEN // 2
MIN_CONFIDENCE = 0.30 # below this, a reading is reported as low confidence
MAX_HR_SPREAD_BPM = 10.0 # inter-quartile spread above this means an unstable capture