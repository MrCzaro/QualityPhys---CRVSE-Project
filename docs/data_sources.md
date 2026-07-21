# CRVSE Data Sources

This document records the dataset boundary for the QualityPhys / CRVSE project.
It is not a redistribution grant. Each source dataset may have its own license,
citation requirements, access rules, or redistribution limits.

## Summary

The project used four rPPG datasets:

| Dataset | Project role | Current app interpretation |
| --- | --- | --- |
| UBFC-rPPG | Clean controlled rPPG dataset | App-relevant still/seated evidence. |
| UBFC-Phys | Facial video with physiological reference signals | App-relevant but harder than UBFC-rPPG. |
| MCD-rPPG | Multi-camera and multimodal rPPG dataset | App-relevant evidence, including frontal-camera and multicamera preprocessing work. |
| ECG-Fitness | Exercise/high-motion/high-HR stress dataset | Archived stress-test evidence. Not included in the current still/seated app support claim. |

## Dataset Roles

### UBFC-rPPG

UBFC-rPPG was used as a controlled webcam-style rPPG dataset. It is one of the
cleanest datasets in the project and was useful for early preprocessing,
ensemble rPPG construction, and model evaluation.

Project artifacts include:

- `Data/processing_log_ubfc_rppg.csv`
- `Data/processing_log_ubfc_rppg_ensemble.csv`
- UBFC-rPPG rows in the live-compatible manifest and baseline summaries

### UBFC-Phys

UBFC-Phys was used to test behavior on a more difficult facial-video dataset
with physiological reference signals. It remained app-relevant, but performance
was weaker than on UBFC-rPPG.

Project artifacts include:

- `Data/processing_log_ubfc_phys.csv`
- `Data/processing_log_ubfc_phys_ensemble.csv`
- UBFC-Phys rows in the live-compatible manifest and baseline summaries

### MCD-rPPG

MCD-rPPG was used for a larger and more varied rPPG corpus. The project includes
single-camera, frontal-camera ensemble, and multicamera preprocessing work.

Project artifacts include:

- `Data/processing_log_mcd_rppg.csv`
- `Data/processing_log_mcd_rppg_ensemble.csv`
- `Data/processing_log_mcd_rppg_multicam.csv`
- MCD-rPPG rows in the live-compatible manifest and baseline summaries

### ECG-Fitness

ECG-Fitness was used to stress-test exercise, high-motion, and high-HR behavior.
It exposed important limitations of both learned and spectral approaches in this
project.

The current app scope does not claim ECG-Fitness robustness. NB13 deliberately
excluded ECG-Fitness from app-relevant selection because the intended demo scope
is still/seated webcam rPPG.

Project artifacts include:

- `Data/processing_log_ecg_fitness.csv`
- `Data/processing_log_ecg_fitness_ensemble.csv`
- ECG-Fitness evidence in earlier ensemble and live-compatible experiments

## Derived Artifacts

The `Data/` directory contains derived logs and audit CSV files, including:

- per-dataset processing logs
- ensemble processing logs
- live-compatible window audit tables
- live-compatible fine-tuning manifest
- frozen baseline predictions and summaries

These artifacts are useful for reproducibility and review, but they should not
be interpreted as raw dataset redistribution.

## Face And ROI Extraction

The preprocessing notebooks and live app use MediaPipe Face Landmarker or
MediaPipe face/landmark tooling for face localization and facial ROI extraction.
The live app expects a Face Landmarker task asset under:

```text
app/live_hr_demo/models/mediapipe/face_landmarker.task
```

That asset is a third-party model file and is not covered by the project
Apache-2.0 license unless its own license allows it.

## Licensing Boundary

The repository Apache-2.0 license is intended for code and documentation owned
by this project.

It does not automatically license:

- raw UBFC-rPPG data
- raw UBFC-Phys data
- raw MCD-rPPG data
- raw ECG-Fitness data
- HDF5 corpora derived from restricted datasets
- pretrained checkpoints from third parties
- MediaPipe model assets
- downloaded models, notebooks, or data from other projects

Before publishing, sharing, or packaging any data artifact, check the original
dataset terms and only include files that are allowed to be redistributed.

## Current App Data Claim

The current live demo should be described as:

```text
still/seated webcam rPPG research demo
```

It should not be described as:

```text
exercise monitor
high-motion monitor
high-HR robust monitor
medical or diagnostic device
validated clinical measurement system
```
