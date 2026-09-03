# CRVSE Model Cards

This file holds one card per model the project ships, newest last. Each card is
self-contained: identity, intended use, input contract, training provenance,
evaluation, limitations, safety and attribution. Adding a model means appending a
card, not rewriting the file.

| Card | Model | App | Role | Status |
| --- | --- | --- | --- | --- |
| 1 | CRVSE PhysFormer Multichannel | `app/live_hr_demo/` (Phase 2, frozen) | experimental, secondary to spectral consensus | superseded |
| 2 | PhysNet v2 | `app/live_vitals/` (Phase 3, active) | primary estimate, cross-checked by a classical method | current |

Both models are research demonstrations. Neither is a medical device, and neither
may be used for diagnosis, treatment, triage or monitoring decisions. Each card
restates this alongside its own scope, because a card read on its own must still
carry its own warnings.

Project-wide architecture, reference-implementation and dataset attribution live in
`docs/citations.md`; dataset terms live in `docs/data_sources.md`.

---

## Card 1 — CRVSE PhysFormer Multichannel (Phase 2, `app/live_hr_demo/`)

This card describes the experimental model used by the frozen Phase-2 QualityPhys
live HR demo. That app is retained as documentation of the Phase-2 work and is no
longer developed; the model below is **not** used by `app/live_vitals/`.

### Model Identity

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

### Intended Use

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

### Input Contract

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

### Architecture Parameters

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

### Training Reference

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

### Evaluation Summary

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

### Tracking Evidence 

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

### Acquisition Rate Sensitivity

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

### Later Checkpoint Adoption Work

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

### Live App Behavior

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

### Limitations

Known limitations:

- live camera FPS and backend sampling latency can limit model usability
- low sampling rates are incompatible with the 0.7-3.5 Hz bandpass contract
- lighting, movement, face position, skin reflection, and ROI quality can
  distort rPPG signals
- model performance is weaker on ECG-Fitness and exercise-like conditions
- output clamping to 40-180 bpm does not make the model clinically safe
- single-user manual tests are not validation

#### Shrinkage Toward The Training Corpus Mean

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

#### Calibration Was Tested And Rejected

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

### Safety Statement

This model is for research demonstration only. It is not a medical device and
must not be used for diagnosis, treatment, triage, or monitoring decisions.

### Attribution

The CRVSE PhysFormer architecture is an original implementation derived from the
published PhysFormer design:

> Z. Yu, Y. Shen, J. Shi, H. Zhao, P. Torr, G. Zhao. "PhysFormer: Facial
> Video-based Physiological Measurement with Temporal Difference Transformer."
> *CVPR*, 2022.

No third-party model code is redistributed in this repository. Architecture,
reference-implementation and dataset attributions for the whole project are
collected in `docs/citations.md`; dataset terms are in `docs/data_sources.md`.

The Phase-3 frame-based model used by `app/live_vitals/` (PhysNet, Yu et al.,
BMVC 2019) is a separate model and is not covered by this card. It has its own,
below: see Card 2.

---

## Card 2 — PhysNet v2 (Phase 3, `app/live_vitals/`)

This card covers the frame-based model used by the Phase-3 app. It is a different
model, a different input contract and a different codebase from Card 1; nothing in
Card 1 transfers to it.

### Model Identity

```text
name: hr_physnet_v2
display name: PhysNet v2 (SNR-loss baseline)
architecture: PhysNet (3D-CNN encoder-decoder)
checkpoint: Data/Phase3/phase_3_physnet_v2_baseline_best.zip
size: 3.11 MB, 79 state-dict tensors
sha256: f30fae250fb948e2... (first 16 hex)
parameters: 768,577 (0.769 M, all trainable)
task: BVP waveform reconstruction; heart rate read out from the waveform
unit: bpm
weights: https://huggingface.co/MrCzaro/hr_physnet_v2
weights licence: CC BY-NC-SA 4.0 (not the repository Apache-2.0)
```

The checkpoint is not committed to this repository (`.gitignore` excludes
`Data/Phase3/*.zip`). It is published separately under **CC BY-NC-SA 4.0**, which is
inherited from DLCN, the most restrictive of the three training corpora — see the
Trained model weights section of `docs/data_sources.md` for why the weights carry
different terms from the code.

Unlike Card 1, this model is the **primary** estimate in its app. The classical
spectral estimator (`hr_spectral`) runs alongside it as an independent cross-check,
not as the headline number.

### Intended Use

Intended use:

- research and portfolio demonstration
- seated, still, front-facing webcam or phone capture in reasonable light
- resting heart rate in roughly the 55-125 bpm range
- comparison against a classical spectral estimate on the same capture

Not intended for:

- diagnosis, treatment, triage or monitoring decisions
- exercise, motion or elevated-HR conditions, which are untested
- heart-rate **variability**, which was measured and found out of reach
- skin tones outside the training corpora, which have not been characterised
- respiratory rate or blood pressure, which the app shows as unavailable

### Input Contract

Every constant below is fixed by how the model was trained. Changing any of them
moves inference off-distribution, which is why they live in one file
(`app/live_vitals/config.py`) rather than being spread across call sites.

```text
input tensor: [1, 3, 160, 72, 72] float32
frame size: 72 x 72 RGB, INTER_AREA downsample
face box: square, max(w,h) x (1 + 0.6) padding, from MediaPipe FaceLandmarker
box stability: one median box from 12 sampled frames, frozen for the capture
clip length: 160 frames
frame rate: 30 fps; higher is decimated, lower is gated
normalisation: per-channel z-score over the whole clip
output: 160-sample BVP waveform per clip
```

Heart rate is read from the waveform by FFT with 8x zero-padding and parabolic peak
refinement, searched over 0.66-3.0 Hz. Overlapping windows advance by 80 frames and
the reported value is the **median** of the surviving windows, not a stitched
waveform: separate windows share no phase or scale, so overlap-adding them can
cancel a genuine pulse.

The same `crops_from_video` path serves the CLI, the web app and every validation
script, so preprocessing parity between evaluation and deployment is structural
rather than defended by review.

### Training Reference

Trained on the Phase-3 corpus — **MCD-rPPG, DLCN and UBFC-rPPG** — under a seed-42
subject-wise split: 739 subjects, 148 held out for validation. The split is
reproducible offline from the CSV logs in `Data/`, which is what makes the held-out
evaluations below verifiable without the corpus itself.

Loss was negative-Pearson on the waveform plus a frequency-matching / SNR term.

From the architecture screen (2026-08-03, cross-subject validation, 148 val subjects):

| Model | Params | Best val MAE | r | DLCN | MCD | UBFC-rPPG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RhythmMamba | 4.92 M | 6.83 | 0.57 | 2.32 | 10.49 | 4.19 |
| **PhysNet** | **0.77 M** | **7.31** | 0.59 | 3.24 | 10.69 | 2.80 |
| PhysFormer | 7.38 M | 9.71 | 0.47 | 2.14 | 15.93 | 2.80 |
| TYrPPG | 2.85 M | 10.77 | 0.45 | 7.52 | 13.61 | 2.80 |
| EfficientPhys | 2.16 M | 12.71 | 0.39 | 2.93 | 20.77 | 2.80 |

PhysNet was selected because it is statistically tied with RhythmMamba — 0.48 bpm
apart, inside run-to-run noise — at **one sixth of the parameters**, which decides
an app that must load and run on a laptop and a phone.

**The shipped checkpoint is the `v2 / SNR-loss` retrain**, recorded in
`Notebooks/Phase 3 Notebooks/NB_P3_18_HR_PhysNet_v2.ipynb`. It replaces the screen's
frequency-matching term with an SNR loss and trains on 1603 recordings (MCD 959,
DLCN 609, UBFC-rPPG 35) at 3206 clips per epoch. Best cross-subject validation MAE
**5.22 bpm at epoch 27**; the `..._baseline_best.zip` in the file name refers to that
run, as distinct from the rejected variant below.

| Dataset | v1 (screen, neg-Pearson + freq) | v2 (neg-Pearson + SNR) | Change |
| --- | ---: | ---: | ---: |
| MCD-rPPG | 10.69 | **6.84** | -3.85 |
| DLCN | 3.24 | **3.08** | -0.16 |
| UBFC-rPPG | 2.80 | 3.13 | +0.33 |
| shared-3 (checkpoint metric) | 7.31 | **5.22** | -2.09 |
| PhysDrive (zero-shot) | not run | 27.79 | — |

The SNR loss is what moved MCD-rPPG, the corpus that had been the ceiling for every
architecture screened: 10.69 down to 6.84 bpm. The same substitution on PhysFormer
moved its MCD figure from 15.93 to 7.70 (NB_P3_20), so the gain belongs to the loss
rather than to one architecture.

Validation agreement for the shipped run, 475 windows:

| Dataset | n | MAE | RMSE | Bias | 95% limits of agreement | Pearson r |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| DLCN | 168 | 3.08 | 9.19 | +0.40 | -17.64 to +18.44 | 0.80 |
| MCD-rPPG | 232 | 6.83 | 15.05 | +0.45 | -29.10 to +30.01 | 0.54 |
| UBFC-rPPG | 7 | 3.13 | 5.86 | -3.13 | -13.63 to +7.36 | 0.97 |
| PhysDrive (zero-shot) | 68 | 27.79 | 38.20 | +24.82 | -32.53 to +82.16 | 0.11 |
| all | 475 | 8.45 | 18.71 | +3.87 | -32.04 to +39.78 | 0.42 |

**A second variant was trained and rejected.** Adding 198 PhysDrive recordings to the
training mix (`plus_physdrive`, 1801 recordings) improved PhysDrive itself from 27.79
to 19.19 bpm but collapsed everything else: DLCN 3.08 to 11.52, MCD 6.84 to 13.98,
UBFC-rPPG 3.13 to 22.61, with per-dataset Pearson r falling to roughly zero across the
board. It was not adopted. The `baseline` run is what ships.

### Evaluation

All figures below come from scripts in `app/live_vitals/scripts/`, run on subjects
the seed-42 split places in validation.

**Held-out UBFC-rPPG, 8 subjects.** `check_ubfc_regression.py`, against a committed
baseline (`ubfc_baseline_hr_physnet_v2.json`) so drift is detected rather than
argued about. Per-window predicted HR is compared against the same spectral readout
applied to the reference BVP over the same windows.

| Subject | Reported | Reference | Error | Window MAE | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| 11 | 120.18 | 123.33 | -3.15 | 2.32 | ok |
| 13 | 108.40 | 108.63 | -0.23 | 1.04 | ok |
| 24 | 106.32 | 108.98 | -2.66 | 2.38 | unstable |
| 25 | 89.02 | 90.43 | -1.41 | 1.46 | ok |
| 34 | 115.87 | 115.80 | +0.07 | 0.43 | ok |
| 35 | 109.17 | 108.23 | +0.95 | 0.70 | ok |
| 42 | 97.30 | 96.96 | +0.34 | 0.43 | ok |
| 47 | 111.24 | 111.41 | -0.17 | 1.16 | ok |
| **mean** | | | **-0.784** | **1.239** | |

**Held-out MCD-rPPG, 6 subjects across 12 recordings** (rest and post-exercise).
Ground truth is the 100 Hz contact PPG that ships frame-aligned with each recording.

```text
per-window HR RMSE:     5.29 bpm
capture-level MAE:      0.32 bpm   (median over ~65 windows per recording)
recordings reporting:   12 / 12
```

For context, the MCD-rPPG paper (Egorov et al., ACMMM 2025) reports frontal-camera
HR MAE of 2.82 (RhythmFormer), 3.80 (POS) and 4.08 (PhysFormer). **These are not
directly comparable**: their metric is MAE per 10-second segment, ours is either a
5.36-second window or a 3-minute median. The comparison is indicative only; a
like-for-like run under their protocol has not been done.

**Independent device check, 2026-09-01.** One resting capture on a laptop webcam
against a pulse oximeter reading 60 bpm: model 61.2, CHROM 61.2, POS 60.9, GREEN
60.6, with 21 of 21 windows kept. All four sit inside the oximeter's own display
tolerance. This is one subject, one session, one lighting condition, and is not
validation.

**Calibration.** The shrinkage-toward-the-corpus-mean failure documented in Card 1 is
**absent** here. Validation bias is +0.40 bpm on DLCN and +0.45 on MCD-rPPG, and on the
MCD subset above it is near zero across 63-96 bpm (+0.04 at 29.9 fps, -0.19 at 24 fps).

The one consistent negative is UBFC-rPPG: -3.13 bpm bias in validation, and -0.784 mean
signed error across the 8 held-out subjects. That corpus sits at 90-123 bpm, the top of
this app's stated range, and the largest single held-out error is -3.15 bpm on subject
11 at 123 bpm — the fastest in the set. High rates are under-read.

### Refusal Behaviour

The model does not always answer, and that is deliberate: a qualified wrong number
is worse than a refusal. Every threshold below was set by measurement.

```text
band-edge rejection      a spectral peak pinned to the search-band edge is not a
                         peak, so the window returns no rate at all
confidence gate          windows below 0.65 spectral concentration are discarded
MAD outlier rejection    windows beyond 3 MADs from the median are discarded
minimum usable windows   fewer than 3 surviving windows is not a reading
minimum usable fraction  below 0.25 surviving, no value is reported at all
acquisition rate         below 20 fps refused outright, below 27 fps flagged
framing                  a box clamped by more than 2% of its side is refused
frame aspect             outside 1.20-1.90 the capture is flagged off-distribution
```

Statuses surfaced to the user: `ok`, `degraded_capture`, `unstable`,
`insufficient_quality`, `insufficient_frames`, `no_estimate`, `capture_rejected`.
The app never shows a value without its status and confidence beside it.

Acquisition rate is a **hard** gate because spectral confidence cannot detect a
wrong time base: a 10 fps webcam capture once produced a confident reading about
7 bpm below a known resting heart rate.

### Limitations

- **Heart-rate variability is out of reach with this model.** Measured across 12
  held-out MCD recordings: 93% of reference beats are found, but median beat-timing
  jitter is 56 ms while the RMSSD being measured is 20-27 ms. The noise is two to
  three times the signal, and RMSSD is built from successive differences, so it
  amplifies precisely that error. This is a property of the reconstruction, not of
  the camera: 30 fps sampling costs only 2% of RMSSD when beat positions are refined
  sub-sample.
- **MCD-rPPG is the standing accuracy ceiling** — around 10 bpm at training time
  against 2-3 on DLCN and UBFC-rPPG, for every architecture screened. Two candidate
  explanations, source frame rate and crop aspect, were each tested and eliminated.
  It remains unexplained.
- **The MCD subset used above was chosen by highest cardiac SQI**, so it is the
  cleanest available and its figures must not be quoted as MCD performance.
- **In-vehicle conditions fail outright.** Zero-shot on PhysDrive the checkpoint scores
  27.79 bpm MAE with +24.82 bpm bias and Pearson r of 0.11 — it is not tracking the
  reference at all. Training on PhysDrive repaired that corpus while destroying the
  other three, so no version of this model is usable in a moving vehicle.
- **Elevated heart rate is untested, and the trend is unfavourable.** The MCD
  post-exercise recordings begin after the subject has settled, so they do not serve as
  an exertion test. What evidence exists points one way: UBFC-rPPG is the fastest corpus
  at 90-123 bpm and carries the only negative validation bias, and the largest held-out
  error is on its fastest subject.
- **Skin tone has not been characterised.** No stratified evaluation has been run,
  and rPPG is known to degrade on darker skin. This is an unquantified gap, not an
  absence of risk.
- **Motion, lighting and backlighting** degrade the signal. The framing gate catches
  gross faults of position but not a badly lit face.
- **The face box is frozen for the capture**, matching training. A subject who moves
  substantially leaves their own crop.
- **Single-operator testing is not validation.** The device check above is one
  person in one session.

### Safety Statement

Research demonstration only. Not a medical device. Must not be used for diagnosis,
treatment, triage or monitoring decisions.

No capture is persisted: the uploaded clip is written to a temporary file that is
deleted under every outcome, and no frame, crop or clip is stored. The interface
states this while the camera is live.

### Attribution

`app/live_vitals/models/architectures/physnet.py` is an original implementation of
the published PhysNet architecture:

> Z. Yu, X. Li, G. Zhao. "Remote Photoplethysmograph Signal Measurement from Facial
> Videos Using Spatio-Temporal Networks." *BMVC*, 2019.

The classical cross-check implements POS (Wang et al., IEEE TBME 2017) and CHROM
(de Haan & Jeanne, IEEE TBME 2013), likewise reimplemented from the papers. No
third-party model code is redistributed in this repository. Project-wide attribution
is collected in `docs/citations.md`; dataset terms are in `docs/data_sources.md`.
