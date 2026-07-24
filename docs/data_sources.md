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

## Required Citations

Each dataset below requires attribution under its own terms. Any publication,
report, presentation, or public artifact derived from this project must cite the
datasets it used. Citation requirements are independent of redistribution
limits: even where a dataset permits derived work, the citations below remain
required.

### UBFC-rPPG

> S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois.
> "Unsupervised skin tissue segmentation for remote photoplethysmography."
> *Pattern Recognition Letters*, 124:82-90, 2019.
> doi:10.1016/j.patrec.2017.10.017

Note on the year: the DOI string contains `2017` because the article was
accepted in 2017, but the volume and page numbers are 2019. Cite 2019.

### UBFC-Phys

Primary publication:

> R. Meziati Sabour, Y. Benezeth, P. De Oliveira, J. Chappe, F. Yang.
> "UBFC-Phys: A Multimodal Database For Psychophysiological Studies Of Social
> Stress." *IEEE Transactions on Affective Computing*, 14(1):622-636, 2021.
> doi:10.1109/TAFFC.2021.3056960

Dataset record (cite alongside the paper, matching the download source):

> Y. Benezeth, R. Meziati Sabour, P. De Oliveira, J. Chappe, F. Yang (2021).
> *UBFC-Phys: A Multimodal Dataset For Psychophysiological Studies Of Social
> Stress.* dataUBFC. doi:10.25666/dataubfc-2022-05-05

The IEEE DataPort distribution carries its own DOI: 10.21227/5da0-7344.

### MCD-rPPG

> K. Egorov, S. Botman, P. Blinov, G. Zubkova, A. Ivaschenko, A. Kolsanov,
> A. Savchenko. "Gaze into the Heart: A Multi-View Video Dataset for rPPG and
> Health Biomarkers Estimation." *Proceedings of the 33rd ACM International
> Conference on Multimedia (ACM MM)*, 2025. arXiv:2508.17924

Affiliations: Sber AI Lab (Moscow), Samara State Medical University, ISP RAS
Research Center for Trusted Artificial Intelligence.

Distribution: https://huggingface.co/datasets/kyegorov/mcd_rppg
Reference code: https://github.com/ksyegorov/mcd_rppg

### ECG-Fitness

> R. Spetlik, V. Franc, J. Cech, J. Matas.
> "Visual Heart Rate Estimation with Convolutional Neural Network."
> *Proceedings of the British Machine Vision Conference (BMVC)*,
> Newcastle, UK, 2018.

Author order follows the citation requested on the dataset distribution page at
the Center for Machine Perception, Czech Technical University in Prague. Other
orderings appear in third-party reference lists; use the order above.

## Dataset Roles

### UBFC-rPPG

UBFC-rPPG was used as a controlled webcam-style rPPG dataset. It is one of the
cleanest datasets in the project and was useful for early preprocessing,
ensemble rPPG construction, and model evaluation.

Citation required: Bobbia et al., *Pattern Recognition Letters*, 2019
(see Required Citations).

Project artifacts include:

- `Data/processing_log_ubfc_rppg.csv`
- `Data/processing_log_ubfc_rppg_ensemble.csv`
- UBFC-rPPG rows in the live-compatible manifest and baseline summaries

### UBFC-Phys

UBFC-Phys was used to test behavior on a more difficult facial-video dataset
with physiological reference signals. It remained app-relevant, but performance
was weaker than on UBFC-rPPG.

Reference signals are contact BVP and electrodermal activity recorded with an
Empatica E4 wristband across a three-stage protocol (rest, speech, arithmetic).

Citation required: Meziati Sabour et al., *IEEE Transactions on Affective
Computing*, 2021 (see Required Citations).

Project artifacts include:

- `Data/processing_log_ubfc_phys.csv`
- `Data/processing_log_ubfc_phys_ensemble.csv`
- UBFC-Phys rows in the live-compatible manifest and baseline summaries

### MCD-rPPG

MCD-rPPG was used for a larger and more varied rPPG corpus. The project includes
single-camera, frontal-camera ensemble, and multicamera preprocessing work.

The dataset provides 3600 recordings from 600 subjects, captured from three
camera sources in resting and post-exercise states, with a 100 Hz reference PPG
signal and 13 additional biomarkers. Respiratory rate is provided as a scalar
clinical measurement per recording, not as a continuous respiratory waveform.

Citation required: Egorov et al., ACM Multimedia, 2025
(see Required Citations).

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

Access route: obtained by signed request form submitted to the Center for
Machine Perception, Czech Technical University in Prague, following the dataset's
stated access procedure. Redistribution is not permitted.

Citation required: Spetlik et al., *BMVC*, 2018 (see Required Citations).

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

Access conditions differ across the four datasets. UBFC-rPPG, UBFC-Phys, and
ECG-Fitness were obtained under their respective request or registration
procedures and are not redistributable. MCD-rPPG is the most permissively
licensed source in the project; confirm the current terms on its distribution
page before relying on that status for any derived release.

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