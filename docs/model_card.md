# CRVSE PhysFormer Model Card

This model card describes the current experimental model used by the
QualityPhys live HR demo.

## Model Identity

```text
name: crvse_physformer_multichannel_v1
display name: CRVSE PhysFormer Multichannel HR
architecture: CRVSEPhysFormer
checkpoint: CRVSETransformer_Ensemble_physformer_multichannel_best.pt
task: heart-rate regression
unit: bpm
```

The model is loaded by the live demo as an experimental secondary estimate. It
is not the primary HR result shown by the app.

## Intended Use

Intended use:

- research and portfolio demonstration
- still or seated webcam rPPG experiments
- comparison against classical spectral HR
- model-behavior diagnostics during live measurements

Not intended for:

- diagnosis or treatment decisions
- medical monitoring
- exercise monitoring
- high-motion robustness claims
- high-HR robustness claims
- ECG-Fitness-style robustness claims

## Input Contract

The current model expects:

```text
input shape: 3 x 240
channels: POS, CHROM, GREEN
window: 8 seconds
normalization: per-window z-score
target_frames: 240
output clamp: 40 to 180 bpm
```

The live app builds these model inputs from browser-collected facial ROI RGB
summaries. The app uses MediaPipe Face Landmarker for face/ROI detection and
then computes POS, CHROM, and GREEN candidate rPPG signals.

Model-side preprocessing uses:

- forehead, left-cheek, and right-cheek ROI summaries
- POS, CHROM, and GREEN candidate signal construction
- local-buffer preprocessing
- bandpass filtering from 0.7 Hz to 3.5 Hz
- latest 8-second model window resampled to 240 frames

Live preprocessing is closer to the training contract than the first prototype,
but it is still not identical to full-recording offline preprocessing.

## Architecture Parameters

From `app/live_hr_demo/configs/model_specs.yaml`:

```text
cnn_channels: 16
freq_channels: 64
d_model: 80
n_heads: 4
n_layers: 4
dim_feedforward: 256
dropout: 0.11331939348791525
hr_min: 40.0
hr_max: 180.0
max_positional_length: 300
```

The model uses PyTorch in the live app.

## Training Reference

The current checkpoint came from the ensemble Transformer-family model search.
It was trained from scratch as part of the CRVSE model-zoo work on ensemble
POS/CHROM/GREEN rPPG artifacts.

Training reference from the app config:

```text
best_n_epochs: 50
best_val_mae: 6.900240182876587
training/reference datasets:
  - MCD-rPPG
  - UBFC-rPPG
  - UBFC-Phys
  - ECG-Fitness
```

The broader model-zoo work included CRVSENet, InceptionNet, ResNet,
Transformer-family models, PhysFormer-style models, LocalAttention-style
models, and 1D adaptations inspired by video rPPG architectures such as TSCAN
and EfficientPhys.

## Evaluation Summary

The 2026-07 training/preprocessing audit identified the multichannel PhysFormer
as the strongest app checkpoint candidate from the ensemble experiments.

Reported held-out test performance from the audit:

| Metric | Value |
| --- | ---: |
| Window MAE | 6.68 bpm |
| Subject MAE | 3.37 bpm |

Per-dataset window MAE from the same audit:

| Dataset | Window MAE |
| --- | ---: |
| MCD-rPPG | 4.35 bpm |
| UBFC-rPPG | 3.20 bpm |
| UBFC-Phys | 8.70 bpm |
| ECG-Fitness | 20.33 bpm |

Interpretation:

- the model was useful offline on parts of the preprocessed ensemble corpus
- ECG-Fitness remained a major weakness
- live app behavior cannot be treated as equivalent to offline test behavior
- the model remains experimental in the product UI

## Later Checkpoint Adoption Work

NB10-NB13 tested whether the frozen source-FPS checkpoint should be replaced.

Summary:

- NB10 shallow fine-tuning did not justify replacement
- NB11 stronger transfer and scratch training did not produce an adoptable
  checkpoint
- NB12 Optuna-guided transfer search did not produce an adoptable checkpoint
- NB13 app-relevant training without ECG-Fitness produced modest transfer gains,
  but no main candidate passed the predefined adoption policy

Current decision:

```text
keep the frozen source-FPS CRVSE PhysFormer checkpoint
```

## Live App Behavior

The live app should expose model uncertainty rather than hide it.

Important states:

- `ok`: model returned a numeric HR estimate
- `rejected`: the model-window quality gate rejected the input
- `skipped`: model inference was not run because a guardrail failed, such as low
  source sampling FPS
- `unavailable`: model path could not return a usable prediction
- `disagrees`: model HR differs materially from model-window spectral HR

The primary app result remains the full-buffer spectral consensus HR from GREEN,
POS, and CHROM.

## Limitations

Known limitations:

- live camera FPS and backend sampling latency can limit model usability
- low sampling rates are incompatible with the 0.7-3.5 Hz bandpass contract
- lighting, movement, face position, skin reflection, and ROI quality can
  distort rPPG signals
- model performance is weaker on ECG-Fitness and exercise-like conditions
- the model can show bias relative to pulse-oximeter spot checks
- output clamping to 40-180 bpm does not make the model clinically safe
- single-user manual tests are not validation

## Safety Statement

This model is for research demonstration only. It is not a medical device and
must not be used for diagnosis, treatment, triage, or monitoring decisions.
