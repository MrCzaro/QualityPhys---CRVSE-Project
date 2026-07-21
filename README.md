# QualityPhys / CRVSE

QualityPhys / CRVSE is a research-stage project for camera-based remote vital
sign estimation from facial video. CRVSE stands for Camera Remote Vital Signs
Estimator. The current working product is a live browser demo for heart-rate
estimation from facial remote photoplethysmography (rPPG).

The project is not a medical device, not a diagnostic tool, and not validated
for clinical decision-making.

## Current State

The live demo estimates heart rate from webcam-derived facial ROI color signals.
Its primary heart-rate estimate is classical spectral consensus from GREEN, POS,
and CHROM candidate rPPG signals. The CRVSE PhysFormer model is shown as an
experimental secondary estimate and is not allowed to silently override the
spectral estimate.

Current supported demo scope:

- still or seated webcam rPPG measurement
- frontal face position
- short measurement windows
- spectral HR as the primary app estimate
- experimental CRVSE model HR as a comparison
- desktop diagnostics for research and debugging
- simplified mobile workflow over local HTTPS

Current unsupported scope:

- exercise monitoring
- high-motion robustness
- general high-HR robustness
- ECG-Fitness-style robustness
- medical validation
- diagnosis or treatment decisions

## Project Goals

The project started as an attempt to build a complete rPPG research pipeline:
dataset preprocessing, signal extraction, model training, live inference, and
honest demo presentation.

The main learning and engineering goals were:

- extract facial ROI time series from public rPPG datasets
- build POS, CHROM, GREEN, and ensemble rPPG representations
- compare classical spectral HR estimates with learned models
- train and audit 1D and Transformer-family models
- understand failure modes caused by motion, domain shift, high HR, and live
  preprocessing mismatch
- build a usable FastHTML and MonsterUI live demo without hiding uncertainty
- keep the scientific claim narrow and evidence-based

## Repository Map

```text
Notebooks/                         Research and preprocessing notebooks
Data/                              Processing logs and derived CSV audit artifacts
app/live_hr_demo/                  FastHTML live HR demo application
app/live_hr_demo/README.md         App-specific run, HTTPS, route, and UI notes
docs/notebook_index.md             Notebook chronology and research conclusions
docs/data_sources.md               Dataset provenance and licensing boundaries
docs/model_card.md                 Current model scope, inputs, limits, and evidence
LICENSE                            Apache-2.0 license for project code and docs
AGENTS.md                          Local collaboration rules; may remain ignored
```

Large datasets, HDF5 corpora, pretrained checkpoints, MediaPipe model assets,
and local HTTPS certificates are not treated as normal source files. They may be
ignored locally and may have separate terms.

## Data Sources

The project work used four rPPG datasets:

- UBFC-rPPG
- UBFC-Phys
- MCD-rPPG
- ECG-Fitness

UBFC-rPPG, UBFC-Phys, and MCD-rPPG define the current app-relevant still/seated
scope. ECG-Fitness was valuable as a stress-test dataset for exercise,
high-motion, and high-HR behavior, but it is not part of the current live demo
support claim.

Raw datasets are external materials. The repository license does not grant
rights to redistribute them. See `docs/data_sources.md` for the dataset boundary
and artifact notes.

## Research Arc

The project moved through five main stages:

1. Dataset preprocessing notebooks created per-dataset ROI and rPPG artifacts.
2. POS-only model experiments compared 1D CNN, Inception, ResNet, Transformer,
   PhysFormer, and transfer-adaptation ideas on simpler signal inputs.
3. Ensemble rPPG experiments compared one-channel ensemble inputs against
   multichannel POS/CHROM/GREEN inputs.
4. Live-compatible notebooks audited whether the offline model could be adapted
   to the live app preprocessing and source-FPS contract.
5. The FastHTML live app was polished into a simple desktop/mobile research demo
   with spectral HR primary and model HR experimental.

The strongest app checkpoint remains the frozen source-FPS multichannel
CRVSE PhysFormer checkpoint. Later NB10-NB13 experiments tested transfer
learning and scratch retraining, but no candidate was adopted as a replacement.

See `docs/notebook_index.md` for a concise chronology.

## Current Model

The app model is:

```text
crvse_physformer_multichannel_v1
architecture: CRVSEPhysFormer
input: POS, CHROM, GREEN
shape: 3 x 240
window: 8 seconds
training/reference datasets: MCD-rPPG, UBFC-rPPG, UBFC-Phys, ECG-Fitness
```

The current checkpoint came from the ensemble Transformer-family model search.
It is treated as an experimental model in the app. The live app displays model
agreement, rejection, skipped, unavailable, and disagreement states explicitly.

See `docs/model_card.md` for the current model card.

## Live Demo

The live demo is in:

```text
app/live_hr_demo/
```

The app uses:

- FastHTML and MonsterUI for the UI shell and server-rendered partials
- browser JavaScript for camera access, timers, sampling, waveform drawing, and
  interaction state
- Python backend routes for frame decoding, MediaPipe Face Landmarker face/ROI
  detection, rPPG signal extraction, spectral analysis, quality gates, and model
  inference
- PyTorch for CRVSE PhysFormer inference

Run locally from the repository root:

```powershell
.\venv\Scripts\python.exe app\live_hr_demo\app.py
```

Install the live app dependencies with:

```powershell
py -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r app/live_hr_demo/requirements.txt
```

The app requirements file is intentionally scoped to `app/live_hr_demo`.
Notebook, Kaggle training, HDF5 audit, and dataset-preprocessing dependencies
are not included there.

If local HTTPS certificates exist at
`app/live_hr_demo/certs/qualityphys-local.pem` and
`app/live_hr_demo/certs/qualityphys-local-key.pem`, the app automatically serves
HTTPS for local network mobile testing. Otherwise it serves HTTP for local
desktop testing.

For app-specific run commands, HTTPS setup, route ownership, mobile behavior,
manual spot-check notes, and smoke tests, see:

```text
app/live_hr_demo/README.md
```

## Verification

Useful app checks:

```powershell
.\venv\Scripts\python.exe app\live_hr_demo\scripts\run_smoke_tests.py
node --check app\live_hr_demo\static\live_demo.js
```

Smoke tests are not camera validation. The live camera workflow still requires
manual browser testing on desktop and mobile.

## Licensing

The repository `LICENSE` file contains Apache License 2.0 text with project
copyright attribution.

The Apache-2.0 license is intended to cover code and documentation owned by this
project. It does not automatically cover:

- raw datasets
- derived datasets if their source terms restrict redistribution
- pretrained checkpoints from third parties
- MediaPipe Face Landmarker model assets
- local certificates
- any other third-party materials

Those materials may have separate licenses, terms, citation requirements, or
redistribution limits. A `NOTICE` file is not included at this stage because no
required bundled third-party NOTICE attribution was identified during this
documentation pass. Add one before distribution if a dependency, model asset, or
redistributed third-party artifact requires it.

## Limitations

The current evidence supports CRVSE as a research and portfolio demo, not as a
validated physiological measurement product.

Known limits:

- live camera acquisition quality depends on lighting, face position, motion,
  browser behavior, and sampling rate
- spectral estimates can fail when channel SQI is weak or inconsistent
- the model can disagree with spectral HR and shows positive bias in some manual
  spot checks
- ECG-Fitness and exercise-like conditions remain outside the app claim
- notebook results are saved research evidence unless explicitly rerun
- pulse oximeter spot checks are useful sanity checks, not formal validation

The current product stance is intentionally conservative: use spectral consensus
as the primary estimate, show the model as experimental, and expose uncertainty
rather than hiding it.
