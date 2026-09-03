# QualityPhys / CRVSE

QualityPhys / CRVSE is a research-stage project for camera-based remote vital sign
estimation from facial video. CRVSE stands for Camera Remote Vital Signs Estimator.
The current deliverable is `app/live_vitals/`, a browser-based demo that measures
heart rate from remote photoplethysmography (rPPG) using a trained frame-based model,
cross-checked against a classical method that shares none of its assumptions.

The project is not a medical device, not a diagnostic tool, and not validated for
clinical decision-making.

## Evidence at a Glance

Held-out subjects only, under the seed-42 subject-wise split, measured end to end
through the same code path the app uses.

| Evaluation | Result |
|---|---|
| UBFC-rPPG, 8 held-out subjects | **1.24 bpm** mean window MAE, −0.78 signed |
| MCD-rPPG, 6 held-out subjects / 12 recordings | **5.29 bpm** window RMSE, 12/12 reporting |
| Cross-subject validation during training | **5.22 bpm** shared MAE (DLCN 3.08, MCD 6.84, UBFC 3.13) |
| Pulse oximeter, one resting capture | oximeter 60 → model 61.2, classical 60.6–61.2 |
| Beat timing (for HRV) | 93% of beats found, but **56 ms** jitter against 20–27 ms of RMSSD |

The last row is the honest boundary. Heart rate works; beat-to-beat timing does not,
and the reason is measured rather than assumed - see *What It Cannot Do*.

## What the App Does

`app/live_vitals/` captures in the browser and analyses in Python. One codebase serves
laptop and phone: the browser records a 60-second clip with `getUserMedia` and
`MediaRecorder` and uploads it once; the server runs the same `crops_from_video` path
used by every validation script, so preprocessing parity between evaluation and
deployment is structural rather than defended by review.

The interface reports a heart rate with its confidence and status, an independent
classical cross-check, a per-window trend, and a diagnostics panel holding every number
behind the reading. HRV and respiratory rate are shown as unavailable rather than
hidden, so the intended scope is visible.

**No capture is stored.** The uploaded clip goes to a temporary file deleted under every
outcome; no frame, crop or clip is persisted. The interface says so while the camera is
live.

### It refuses to answer

A qualified wrong number is worse than no number. Every threshold below was set by
measurement, not taste:

| Gate | Behaviour |
|---|---|
| Band-edge peak | A spectral peak on the search-band edge is not a peak - window discarded |
| Confidence | Windows below 0.65 spectral concentration discarded |
| MAD outliers | Windows beyond 3 MADs from the median discarded |
| Usable windows | Fewer than 3 survivors is not a reading |
| Usable fraction | Below 0.25 surviving, no value is reported |
| Acquisition rate | Below 20 fps refused outright; below 27 fps flagged |
| Framing | A face box clamped by more than 2% of its side is refused |

Acquisition rate is a hard gate because spectral confidence cannot detect a wrong time
base: a 10 fps capture once produced a confident reading about 7 bpm below a known
resting heart rate.

### Two estimators, deliberately not blended

The neural model is primary. `hr_spectral` runs beside it - POS, CHROM and GREEN, fixed
linear projections with no learned parameters - as a cross-check that fails on different
things. Where they agree, that is corroboration rather than a model confirming itself.

Blending them was tested and rejected. POS and CHROM per-window errors correlate at
r = +0.945 on UBFC, so averaging them buys ~1.4% and measures nothing; averaging the
model with a classical estimate costs a factor of six on MCD. The value is in the
disagreement, which the diagnostics panel surfaces.

## The Model

```text
name:        hr_physnet_v2
architecture: PhysNet 3D-CNN encoder-decoder (Yu et al., BMVC 2019)
parameters:  768,577 (3.11 MB)
input:       [1, 3, 160, 72, 72] — 160 frames of a 72×72 face crop at 30 fps
output:      160-sample BVP waveform; heart rate read out spectrally
trained on:  MCD-rPPG, DLCN, UBFC-rPPG
weights:     https://huggingface.co/MrCzaro/hr_physnet_v2  (CC BY-NC-SA 4.0)
```

The checkpoint is not committed here; `docs/model_card.md` Card 2 documents its input
contract, evaluation, refusal behaviour and limitations in full.

### Why this one

Seven architectures were trained on one fixed pipeline - five in the original screen,
with RhythmFormer and PhaseNet added afterwards. Best cross-subject validation MAE on
the shared three corpora, taking each model's strongest run:

| Model | Shared MAE | DLCN | MCD | UBFC-rPPG | Checkpoint |
|---|---:|---:|---:|---:|---:|
| RhythmFormer | **4.26** | 2.14 | 5.77 | 4.69 | 13.5 MB |
| **PhysNet v2** | **5.22** | 3.08 | 6.84 | 3.13 | **3.1 MB** |
| PhysFormer v2 | 5.63 | 2.88 | 7.70 | 3.12 | — |
| PhaseNet (SNR) | 6.15 | 2.88 | 8.56 | 4.69 | — |
| RhythmMamba v2 | 7.50 | 6.56 | 8.30 | 3.12 | — |

TYrPPG and EfficientPhys were dropped at the screen stage - 10.77 and 12.71 shared MAE,
and TYrPPG at roughly 26 minutes per epoch - and were not retrained.

RhythmFormer is the most accurate and is **not** used: it is 4.3× the checkpoint size
and its inference cost does not fit a capture-then-analyse loop on a laptop or phone.
PhysNet gives up 0.96 bpm for that. Size and latency are selection criteria here, not
only accuracy.

### The loss mattered more than the architecture

Replacing a frequency-matching term with an SNR loss moved three of the four
architectures it was tried on, and moved them by more than the gap between
architectures:

| Model | frequency loss | SNR loss |
|---|---:|---:|
| PhysNet | 7.31 | **5.22** |
| PhysFormer | 9.71 | **5.63** |
| PhaseNet | 10.32 | **6.15** |
| RhythmMamba | 6.83 | 7.50 |

PhaseNet is the cleanest evidence: both variants ran in one notebook with everything
else fixed. RhythmMamba is the exception and got slightly worse, so this is a strong
tendency rather than a rule. The gains land hardest on MCD-rPPG, which had been the
ceiling for every architecture screened.

## What It Cannot Do

Each of these was measured, not assumed.

**Heart rate variability.** The model finds 93% of reference beats, but median
beat-timing jitter is 56 ms while resting RMSSD is 20–27 ms. Noise exceeds signal by
two to three times, and RMSSD is built from successive differences, so it amplifies
exactly that error. This is not the frame rate: resampling a contact PPG to 30 Hz costs
only 2% of RMSSD when beat positions are refined sub-sample. Averaging overlapping
model windows was tried and recovers 4%, because their errors correlate at r ≈ 0.83.
The remaining lever is a timing-aware training objective.

**In-vehicle conditions.** Zero-shot on PhysDrive: 27.79 bpm MAE, +24.82 bias,
r = 0.11 - not tracking at all. Training on PhysDrive repaired that corpus and destroyed
the other three.

**Skin tone.** Not characterised. No stratified evaluation has been run and rPPG is
known to degrade on darker skin. An unquantified gap, not an absence of risk.

**Elevated heart rate.** Untested, and the trend is unfavourable: UBFC-rPPG is the
fastest corpus at 90–123 bpm and carries the only negative validation bias.

**MCD-rPPG remains an unexplained ceiling** - roughly 7 bpm against 3 on the other
corpora, for every architecture screened. Frame rate and crop aspect were each tested as
explanations and eliminated.

Also: motion, poor light and backlighting degrade the signal; the face box is frozen per
capture, so a subject who moves leaves their own crop; and single-operator testing is
not validation.

## Running It

```powershell
.\venv\Scripts\python.exe -m app.live_vitals.web.app          # http://localhost:8000
.\venv\Scripts\python.exe -m app.live_vitals.web.app --https  # https://localhost:8443
```

HTTPS is needed for camera access from a phone on the LAN; certificates are read from
`app/live_hr_demo/certs/`.

A command line exists for recorded video and direct webcam capture:

```powershell
.\venv\Scripts\python.exe -m app.live_vitals models
.\venv\Scripts\python.exe -m app.live_vitals analyze path\to\video.mp4
.\venv\Scripts\python.exe -m app.live_vitals camera --seconds 60
```

## Verification

```powershell
.\venv\Scripts\python.exe -m app.live_vitals.scripts.check_ubfc_regression
.\venv\Scripts\python.exe -m app.live_vitals.scripts.check_contract_parity
.\venv\Scripts\python.exe -m app.live_vitals.scripts.check_hrv_sampling_ceiling
.\venv\Scripts\python.exe -m app.live_vitals.scripts.check_hrv_from_video
```

`check_ubfc_regression` compares reported value, window MAE, usable fraction and status
against a committed per-estimator baseline, and fails on drift. The HRV scripts carry a
reference-against-itself control row; any result there other than `0.00 / 100% / 0.0 ms`
means the harness is broken rather than the model.

These require the held-out corpora locally and are not camera validation. The live
capture path still needs manual browser testing.

## Repository Map

```text
app/live_vitals/                   Phase-3 app: frame-based model, web + CLI  (current)
app/live_hr_demo/                  Phase-2 app: 1D POS/CHROM/GREEN demo       (frozen)
Notebooks/Phase 3 Notebooks/       Preprocessing, dataloader, 7 architectures, v2 retrains
Notebooks/Phase 2 Notebooks/       1D signal model search and audits
Data/                              Processing logs and derived audit CSVs
docs/model_card.md                 One card per shipped model; Card 2 is current
docs/data_sources.md               Dataset provenance, licensing boundary, model weights
docs/citations.md                  Architecture, classical-method and dataset citations
docs/notebook_index.md             Notebook chronology and research conclusions
LICENSE                            Apache-2.0, for project code and documentation
```

`app/live_hr_demo/` is retained deliberately as documentation of the Phase-2 work. It is
no longer developed and its model has a characterised calibration failure recorded in
Card 1 of the model card.

Datasets, HDF5 corpora, checkpoints, MediaPipe assets and local certificates are not
normal source files. They are ignored locally and carry separate terms.

## Research Arc

**Phase 1–2** built the pipeline on hand-crafted 1D signals: per-dataset ROI extraction,
POS/CHROM/GREEN construction, then a model search across 1D CNN, Inception, ResNet and
Transformer families. It produced a working demo and one clearly characterised failure -
the model shrank predictions toward a training-corpus mean near 88 bpm, giving a positive
bias for resting adults. Calibration was fitted and rejected: linear de-shrinkage reached
a correct slope at 38% worse MAE.

**Phase 3** moved to frame-based models that consume video directly and reconstruct
waveforms. Three corpora were re-preprocessed into a unified schema, six architectures
were screened under one fixed pipeline, and the SNR-loss retrain produced the current
checkpoint. The shrinkage bias is gone: validation bias is +0.40 bpm on DLCN and +0.45 on
MCD-rPPG.

The app was rebuilt around the frame contract, with the quality gates and the classical
cross-check added on evidence gathered while validating it.

`docs/notebook_index.md` has the chronology.

## Data Sources

| Dataset | Role | Access |
|---|---|---|
| MCD-rPPG | training + held-out evaluation | most permissive source |
| DLCN | training, low-light robustness | CC BY-NC-SA 4.0 (Kaggle release) |
| UBFC-rPPG | training + held-out evaluation | research use, no formal licence |
| PhysDrive | zero-shot in-vehicle benchmark | per request |
| UBFC-Phys, ECG-Fitness | Phase-2 work only | per request / registration |
| VitalVideos | planned | per request, academic and commercial separate |

Raw datasets are external materials and the repository licence grants no rights to
redistribute them. See `docs/data_sources.md` for the boundary, per-dataset terms, and
the reasoning behind the licence on published model weights.

## Licensing

Three artifacts, three positions:

- **Code and documentation** — Apache-2.0, per `LICENSE`.
- **Model weights** — CC BY-NC-SA 4.0, non-commercial. Weights inherit the most
  restrictive training input, which is DLCN. The reasoning is in `docs/data_sources.md`.
- **Datasets, third-party checkpoints, MediaPipe assets, certificates** — separate terms
  in every case; the repository licence does not reach them.

Two points in `docs/citations.md` affect distribution: the rPPG-Toolbox reference
implementation is under a Responsible AI licence whose behavioural restrictions include
diagnosing medical conditions without human oversight, and the PhaseNet reference
implementation states no licence at all. No third-party model code is redistributed here;
the architectures used are original implementations of published designs.

## Standing Position

The evidence supports this as a research and portfolio project, not a validated
measurement product. The stance throughout has been to state the scope narrowly, gate
what cannot be measured reliably, and publish the negative results - the HRV ceiling, the
rejected ensemble, the rejected calibration, the unexplained MCD gap - alongside the
numbers that worked.
