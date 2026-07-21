# QualityPhys Live HR Demo

Browser-based research demo for camera-derived facial rPPG heart-rate
estimation. The app uses a simple live webcam workflow, classical spectral rPPG
analysis, and an experimental CRVSE PhysFormer model output.

This app is a research and portfolio demo. It is not a medical device, not a
diagnostic tool, and not validated for clinical decision-making.

## Current Scope

Supported demo scope:

- still or seated webcam rPPG measurement
- frontal face position
- short measurement windows
- spectral HR as the primary app estimate
- experimental model HR as a secondary diagnostic estimate
- desktop diagnostics for development and research review
- simplified mobile measurement flow over local HTTPS

Not supported by current evidence:

- exercise monitoring
- high-motion robustness
- high-HR robustness
- ECG-Fitness-style exercise conditions
- medical validation
- diagnosis or treatment decisions

## User Workflow

Desktop workflow:

1. Start the camera.
2. Hold a steady frontal face position.
3. Start a 15 second measurement.
4. Review spectral HR, signal quality, model status, and demo readiness.
5. Use advanced diagnostics only when debugging acquisition or backend behavior.

Mobile workflow:

1. Open the HTTPS local-network URL.
2. Tap Start measurement.
3. Allow camera access.
4. Use the 3 second positioning countdown.
5. Hold still for the 15 second measurement.
6. Review the automatically displayed result.

The mobile view intentionally hides advanced diagnostics. It is designed as a
simple demo path, not a research console.

## App Layout Layers

The current app layout separates the main demo result from deeper research
diagnostics.

Primary result layer:

- final interpretation panel shown above detailed result cards
- plain recommendation about whether to use the spectral estimate or reject the
  window
- explicit model-disagreement wording when the experimental model differs from
  model-window spectral HR
- spectral HR remains the primary app estimate

Measurement detail layer:

- collapsible measurement details section
- estimated HR card
- experimental model HR card
- model-versus-spectral agreement card
- signal-quality card

Desktop support layer:

- collapsible demo-readiness checklist
- browser secure-context status
- camera status
- sample count and live model FPS margin
- signal status
- experimental model status

Desktop research diagnostics layer:

- advanced manual acquisition controls
- frame capture and ROI overlay checks
- backend frame and face debug output
- detailed signal and model diagnostic payloads
- repeatability table for repeated predictions in one browser session

Mobile demo layer:

- start measurement opens the camera and begins the guided workflow
- stop measurement stops acquisition, closes the camera, and clears the current
  run
- a 3 second positioning overlay gives the user time to center the face before
  sampling starts
- the waveform is hidden to keep the mobile path simple
- advanced diagnostics are hidden on mobile
- after measurement completion, the page scrolls to the final interpretation

## Scientific Interpretation

The primary heart-rate estimate is the full-buffer spectral consensus from:

- GREEN
- POS
- CHROM

The CRVSE PhysFormer model output is experimental. It can be useful for
comparison, but it must not override the spectral estimate without stronger
validation evidence.

Important states:

- Accepted: enough spectral support exists for the primary estimate.
- Rejected: the signal-quality gate rejected the window.
- Skipped: a guardrail prevented model inference, for example low live sampling
  FPS.
- Disagrees: the model differs materially from model-window spectral HR.
- Unavailable: the model path could not return a usable prediction.

A good signal-quality label does not mean the experimental model is trustworthy.
Signal quality and model agreement are separate concepts.

## Manual Pulse Oximeter Spot-Check

On 2026-07-20, the app was manually compared against a simple FS20C pulse
oximeter. This was a single-user spot-check, not a formal validation study. The
measurements were not waveform-synchronized, the oximeter itself is not treated
as a clinical research reference standard, and push-up runs are outside the
supported still/seated app scope.

The table below uses the app repeatability-table values: experimental model HR
and model-window spectral HR.

| Run | Model HR | Model-window spectral | Model - spectral | GREEN SQI | POS SQI | CHROM SQI | Used s | Samples | FPS | Oximeter | Activity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 90.1 bpm | 75.0 bpm | +15.1 bpm | 0.799 / good | 0.611 / good | 0.583 / good | 7.99 | 148 | 18.69 | 73 | sit |
| 2 | 85.0 bpm | 75.0 bpm | +10.0 bpm | 0.755 / good | 0.609 / good | 0.635 / good | 7.95 | 137 | 18.37 | 74 | sit |
| 3 | 92.1 bpm | 82.5 bpm | +9.6 bpm | 0.384 / moderate | 0.268 / poor | 0.212 / poor | 7.97 | 142 | 18.55 | 88 | after 10 push ups |
| 4 | 93.9 bpm | 82.5 bpm | +11.4 bpm | 0.716 / good | 0.601 / good | 0.482 / moderate | 7.97 | 143 | 18.18 | 85 | after 10 push ups |
| 5 | 96.4 bpm | 90.0 bpm | +6.4 bpm | 0.627 / good | 0.393 / moderate | 0.388 / moderate | 8.00 | 131 | 16.64 | 85 | after 20 push ups |
| 6 | 96.3 bpm | 82.5 bpm | +13.8 bpm | 0.759 / good | 0.540 / good | 0.505 / good | 8.00 | 137 | 17.54 | 84 | after 20 push ups |
| 7 | 68.9 bpm | 60.0 bpm | +8.9 bpm | 0.785 / good | 0.727 / good | 0.691 / good | 7.96 | 132 | 17.12 | 63 | sit |
| 8 | 76.1 bpm | 60.0 bpm | +16.1 bpm | 0.645 / good | 0.419 / moderate | 0.397 / moderate | 7.96 | 160 | 20.12 | 61 | sit |
| 9 | 101.1 bpm | 82.5 bpm | +18.6 bpm | 0.726 / good | 0.666 / good | 0.623 / good | 7.99 | 136 | 17.36 | 86 | after 20 push ups |
| 10 | 83.3 bpm | 75.0 bpm | +8.3 bpm | 0.688 / good | 0.631 / good | 0.580 / good | 7.95 | 140 | 17.86 | 72 | 1 min post exercise |

Simple descriptive comparison against the oximeter:

| Subset | Model MAE | Model bias | Model-window spectral MAE | Model-window spectral bias |
| --- | --- | --- | --- | --- |
| All 10 runs | 11.22 bpm | +11.22 bpm | 2.80 bpm | -0.60 bpm |
| Sitting only | 12.27 bpm | +12.27 bpm | 1.75 bpm | -0.25 bpm |
| Push-up/post-exercise runs | 10.52 bpm | +10.52 bpm | 3.50 bpm | -0.83 bpm |

Interpretation:

- This spot-check supports keeping spectral HR as the primary app estimate.
- The experimental model showed a consistent positive bias in these runs.
- The model-window spectral estimate was much closer to the oximeter in this
  small manual sample.
- Push-up and post-exercise rows are useful stress observations, but they do
  not expand the supported app scope beyond still/seated webcam rPPG.

## Live Demo Route Ownership

The live app uses separate routes for measurement, diagnostics, analysis, and
rendered UI partials.

| Route | Owner | Purpose |
| --- | --- | --- |
| `POST /api/roi-sample` | Main and manual ROI sampling | Compact live acquisition route. Receives one browser frame, extracts face ROI RGB summaries, and returns only the fields needed by the live sample buffer. |
| `POST /api/debug-frame` | Advanced diagnostics | Decodes one submitted frame and returns frame-level image diagnostics. |
| `POST /api/debug-face` | Advanced diagnostics | Runs face detection and ROI overlay diagnostics for one captured frame. Used by the desktop "Detect face + draw ROIs" flow. |
| `POST /api/analyze-roi-series` | Signal analysis | Converts browser-collected ROI RGB samples into GREEN, POS, and CHROM candidate rPPG signals and computes spectral summaries. |
| `POST /api/predict-live-roi-series` | Experimental model inference | Builds the model input from ROI samples and runs the CRVSE PhysFormer path when guardrails allow it. |

The main measurement loop should call `/api/roi-sample`, not `/api/debug-face`.
The debug face route remains available for desktop advanced diagnostics.

Large server-rendered diagnostic payloads use POST JSON UI routes, including:

- `POST /ui/roi-analysis-summary-json`
- `POST /ui/model-prediction-summary-json`
- `POST /ui/repeatability-table-json`

## Frontend And Backend Responsibilities

FastHTML and MonsterUI own:

- page structure
- cards and panels
- result summaries
- diagnostic partials
- desktop/mobile presentation scaffolding

JavaScript owns:

- browser camera access
- local video preview
- measurement timers
- 3 second mobile positioning countdown
- completion-driven ROI sampling
- waveform drawing
- browser-side sample buffer
- calls to backend APIs and server-rendered partials

Python backend owns:

- image decoding
- MediaPipe face detection
- ROI construction
- ROI RGB summaries
- rPPG signal extraction
- spectral analysis
- quality gates
- model input preparation
- model inference
- JSON-safe API payloads

## Privacy Boundary

Current privacy behavior:

- live preview stays in the browser
- submitted frames are processed in backend memory only
- frames are not intentionally stored
- the measurement buffer stores numeric ROI summaries, not raw images
- model prediction uses numeric ROI time series

Do not add raw-frame persistence, uploads, logging, or localStorage storage
without an explicit project decision.

## Local Run

From the repository root:

```powershell
.\venv\Scripts\python.exe app/live_hr_demo/app.py
```

## App Dependencies

The app-specific dependency file is:

```text
app/live_hr_demo/requirements.txt
```

Create and install a local virtual environment from the repository root:

```powershell
py -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r app/live_hr_demo/requirements.txt
```

This requirements file is for the live demo app only. It intentionally excludes
notebook, Kaggle training, HDF5 audit, and dataset-preprocessing dependencies
that are not required by the live app runtime.

Default port:

```text
5001
```

The app automatically uses local HTTPS when these files exist:

```text
app/live_hr_demo/certs/qualityphys-local.pem
app/live_hr_demo/certs/qualityphys-local-key.pem
```

The same paths can also be provided explicitly:

```powershell
$env:QUALITYPHYS_HTTPS_CERT="D:\code\QualityPhys - CRVSE Project\app\live_hr_demo\certs\qualityphys-local.pem"
$env:QUALITYPHYS_HTTPS_KEY="D:\code\QualityPhys - CRVSE Project\app\live_hr_demo\certs\qualityphys-local-key.pem"
.\venv\Scripts\python.exe app/live_hr_demo/app.py
```

Desktop URL:

```text
https://localhost:5001/
```

Mobile local-network URL:

```text
https://<laptop-ip>:5001/
```

Mobile camera access generally requires HTTPS.

## Smoke Tests

Run the app smoke-test suite from the repository root:

```powershell
.\venv\Scripts\python.exe app/live_hr_demo/scripts/run_smoke_tests.py
```

Run the JavaScript syntax check after browser-side changes:

```powershell
node --check app/live_hr_demo/static/live_demo.js
```

The smoke suite checks model contracts, runtime fallback payloads, ROI sample
API contract, SQI behavior, window quality, integrated inference, serialization,
and synthetic inference. It does not replace manual browser-camera testing.

