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

## Tracking Evidence 

MAE alone does not show whether a regression model is responding to its input. A
later audit pass added tracking statistics: the OLS slope of predicted HR on
reference HR, and the correlation between them. A slope near 1.0 means the model
follows the reference; a slope near 0 means it emits a near-constant value.

Measured on held-out test subjects with training-style preprocessing
(`stored_reference` mode), using `audit_model_prediction_variance.py`:

| Dataset scope | Windows | Slope | Pearson r | MAE |
| --- | ---: | ---: | ---: | ---: |
| All four datasets | 96 | 0.419 | 0.506 | 12.91 bpm |
| App-relevant three | 48 | 0.904 | 0.906 | 5.05 bpm |

App-relevant means MCD-rPPG, UBFC-rPPG, and UBFC-Phys, which is the still/seated
scope the live demo actually targets.

Interpretation:

- on its intended domain, with its intended preprocessing, the checkpoint tracks
  reference HR closely
- the weaker aggregate figures reported earlier are consistent with ECG-Fitness
  contamination of the evaluation scope rather than with a weak checkpoint
- this is independent support for the NB13 decision to exclude ECG-Fitness from
  app-relevant model selection
- it does not change the product decision: spectral consensus remains the primary
  app estimate and model HR remains experimental

Caveats on this evidence:

- the app-relevant test split here is 48 windows, so the slope estimate carries a
  standard error of roughly 0.06
- a larger all-splits run of 360 windows reproduces the same pattern with tighter
  estimates, which is what makes the result credible
- these are offline HDF5 windows, not live camera measurements
- no result here has been validated against a reference device in the live app

## Acquisition Rate Sensitivity

The same audit measured tracking against simulated acquisition rates, using
app-relevant datasets and training-style local-buffer preprocessing:

| Simulated rate | Slope | Pearson r | MAE |
| --- | ---: | ---: | ---: |
| Source FPS (~30 Hz) | 0.870 | 0.900 | 5.25 bpm |
| 30 Hz | 0.868 | 0.899 | 5.13 bpm |
| 20 Hz | 0.635 | 0.749 | 8.40 bpm |
| 15 Hz | 0.629 | 0.747 | 8.77 bpm |
| 10 Hz | 0.681 | 0.794 | 7.74 bpm |
| 7.5 Hz | 0.589 | 0.389 | 17.37 bpm |

Two practical points:

- the degradation is a **step** between 30 Hz and 20 Hz, then roughly flat from
  20 Hz down to 10 Hz; acquisition work below 30 Hz buys little
- 7.5 Hz fails by noise amplification rather than by flattening, because the
  3.5 Hz bandpass cutoff reaches 0.933 of the Nyquist frequency there and the
  order-4 Butterworth becomes close to degenerate

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
- output clamping to 40-180 bpm does not make the model clinically safe
- single-user manual tests are not validation

### Shrinkage Toward The Training Corpus Mean

This limitation was characterized on 2026-07-21 and replaces the earlier, vaguer note that the model "can show bias relative to pulse-oximeter spot checks".

The training corpus has a mean HR of roughly 88 bpm, because it includes
post-exercise and exercise recordings. The model behaves like a shrinking
regressor, so predictions are pulled toward that corpus mean:

```text
prediction is approximately:
    corpus_mean + slope * (reference_hr - corpus_mean)
```

At an acquisition rate near 20 Hz the measured slope is about 0.635. For a seated
user with a true HR near 63 bpm this predicts roughly 72 bpm. Four live runs in
one session produced 78.0 to 78.9 bpm while spectral consensus reported 60 to
67.5 bpm, which is consistent in direction and approximate magnitude.

Practical consequences:

- the model shows a **positive bias for users whose HR is below the corpus mean**,
  which includes most resting seated adults
- the bias grows as the user's HR moves further below roughly 88 bpm
- because shrinkage compresses variation, model HR can appear nearly constant
  across repeated measurements of the same resting subject
- this is a calibration property, not evidence that the model ignores its input;
  measured slope at 20 Hz is 0.635, not 0

### Calibration Was Tested And Rejected

Linear and offset corrections were fitted on train subjects and evaluated on
held-out test subjects on 2026-07-21. Neither should be adopted.

At an acquisition rate near 20 Hz, on held-out test subjects:

| Correction | MAE | Bias | p90 | Slope |
| --- | ---: | ---: | ---: | ---: |
| none | 8.40 | +4.38 | 19.05 | 0.635 |
| offset | 8.21 | -0.02 | 16.33 | 0.635 |
| linear | 11.57 | -0.52 | 26.61 | 1.080 |

Linear de-shrinkage reaches a calibrated slope but costs roughly 38 percent more
MAE and 40 percent worse p90. This is expected: shrinkage is close to MSE-optimal
for a noisy predictor, so removing it trades bias for variance.

The deeper issue is that the model is unbiased near 92 to 98 bpm, while the live
demo serves seated resting adults near 60 to 75 bpm. For a user with a true HR of
65 bpm the systematic error is about +5.9 bpm at source FPS and +13.6 bpm at
20 Hz. An offset correction removes the average bias measured across a corpus
centred near 88 bpm, which does not help a user at 65 bpm.

The lever for a genuinely better model here would be the **training
distribution**, reweighted toward resting HR, rather than architecture, transfer
learning, or post-hoc calibration. No such work has been done.

Until it is, model HR remains experimental and subordinate to spectral consensus
HR, which performs well in this domain.

## Safety Statement

This model is for research demonstration only. It is not a medical device and
must not be used for diagnosis, treatment, triage, or monitoring decisions.
