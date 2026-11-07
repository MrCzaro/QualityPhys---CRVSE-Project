# CRVSE Notebook Index

This file summarizes the main notebook chronology and conclusions for the
QualityPhys / CRVSE project. It is a navigation and interpretation aid, not a
claim that every notebook was freshly rerun during this documentation pass.

## Reading Guide

The notebooks fall into two phases:

- **Phase 2** (`Notebooks/Phase 2 Notebooks/`): 1D POS/CHROM/GREEN signals and the
  models trained on them - dataset preprocessing, POS-only experiments, ensemble and
  multichannel experiments, and live-compatible adaptation. They produced the Phase-2
  app, `app/live_hr_demo/`, now frozen; its model's characterised failure (predictions
  shrinking toward the corpus mean) is recorded in Card 1 of the model card.
- **Phase 3** (`Notebooks/Phase 3 Notebooks/`, NB_P3_01 to NB_P3_40): models that read
  the face-crop video itself - the unified preprocessing schema, the architecture screen,
  the shipped model (PhysNet v2, NB_P3_18), held-out and benchmark sets, full-corpus
  retraining, the app-level battery, beat timing, robustness checks of the heart-rate
  reading, and respiratory rate from the torso's motion in the whole frame. They are the
  basis of the current app, `app/live_vitals/`, and of Card 2.

Within a phase, later notebooks should be trusted more than earlier ones when they
disagree, because they were written after the issues the earlier ones exposed. Where
both phases speak to the same question, Phase 3 supersedes Phase 2.

## Phase 2: Dataset Preprocessing

| Notebook | Role | Notes |
| --- | --- | --- |
| `NB_P2_01_ECG_Fitness.ipynb` | ECG-Fitness preprocessing | Built the first ECG-Fitness rPPG preprocessing path for an exercise/high-motion dataset. |
| `NB_P2_02_UBFC-rPPG.ipynb` | UBFC-rPPG preprocessing | Prepared the clean controlled UBFC-rPPG dataset. |
| `NB_P2_03_UBFC-Phys.ipynb` | UBFC-Phys preprocessing | Prepared UBFC-Phys facial video and physiological reference signals. |
| `NB_P2_04_MCD-rPPG.ipynb` | MCD-rPPG preprocessing | Prepared MCD-rPPG recordings and metadata for later model work. |
| `NB_P2_05_MCD-rPPG_MultiCam.ipynb` | MCD multi-camera preprocessing | Extended MCD work to multiple camera views. The rRESP and rBCG paths produced near-null SQI, which is an important negative finding for later reference. |
| `NB_P2_06_MCD-rPPG_FrontalCam_Ensemble.ipynb` | MCD frontal-camera ensemble | Created frontal-camera ensemble rPPG artifacts. |
| `NB_P2_07_UBFC-rPPG_Ensemble.ipynb` | UBFC-rPPG ensemble | Built POS, CHROM, and GREEN ensemble-style artifacts for UBFC-rPPG. |
| `NB_P2_08_UBFC-Phys_Ensemble.ipynb` | UBFC-Phys ensemble | Built POS, CHROM, and GREEN ensemble-style artifacts for UBFC-Phys. |
| `NB_P2-09_ECG_Fitness_Ensemble.ipynb` | ECG-Fitness ensemble | Historical ECG-Fitness ensemble artifact. It contains a saved error output and should be treated as a development artifact rather than a clean final notebook. |

## Phase 2: POS-Only Model Experiments

These notebooks explored simpler 1D signal learning before the project moved to
multichannel POS/CHROM/GREEN inputs.

| Notebook | Role | Notes |
| --- | --- | --- |
| `CRVSE POS Optuna Trials CRVSENet1D (Round 1).ipynb` | 1D CNN baseline | Early POS-only CRVSENet1D Optuna experiment. |
| `CRVSE POS Optuna Trials InceptionNet1D (Round 2).ipynb` | Inception-style baseline | POS-only InceptionNet1D search. |
| `CRVSE POS Optuna Trials Transformers (Round 3).ipynb` | Transformer family | POS-only Transformer and PhysFormer-style exploration. |
| `CRVSE POS Optuna Trials ResNet (Round 4).ipynb` | ResNet family | POS-only ResNet and ResNet-SE exploration. |
| `CRVSE POS Optuna Trials Weight Transfer.ipynb` | Pretrained adaptation | Adapted video rPPG model ideas such as TSCAN, EfficientPhys, and PhysFormer into 1D POS experiments. |
| `CRVSE POS Optuna Trials FT Head-Only.ipynb` | Head-only fine-tuning | Tested frozen-backbone, head-only adaptation strategies. |

Main interpretation: POS-only experiments were useful for learning and model-zoo
triage, but they were not the final Phase-2 app contract. The Phase-2 app and its
model used multichannel POS/CHROM/GREEN inputs; the current app reads the face video
itself (Phase 3).

## Phase 2: Ensemble And Multichannel Model Experiments

These notebooks trained model families on rPPG ensemble artifacts. They are the
main source of the Phase-2 app checkpoint.

| Notebook | Role | Notes |
| --- | --- | --- |
| `CRVSE Phase B - Ensemble rPPG - CRVSENET (Round 1).ipynb` | Ensemble model baseline | Tested CRVSENet-style learning on ensemble rPPG. |
| `CRVSE Phase B - Ensemble rPPG - CRVSEInceptionNet 1D (Round 2) part 1 - ensemble only (1D).ipynb` | One-channel Inception | Tested InceptionNet on one-channel ensemble rPPG. |
| `CRVSE Phase B - Ensemble rPPG - CRVSEInceptionNet 1D (Round 2) part 2 - multichannel (3ch).ipynb` | Multichannel Inception | Tested InceptionNet on POS/CHROM/GREEN. |
| `CRVSE Phase B - Ensemble rPPG - CRVSEResNet 1D (Round 3) part 1 - ensemble only (1ch).ipynb` | One-channel ResNet | Tested ResNet on one-channel ensemble rPPG. |
| `CRVSE Phase B - Ensemble rPPG - CRVSEResNet 1D (Round 3) part 2 - multichannel (3ch).ipynb` | Multichannel ResNet | Tested ResNet on POS/CHROM/GREEN. |
| `CRVSE Phase B - Ensemble rPPG - CRVSETransformer family (Round 4) part 1 - ensemble only (1ch).ipynb` | One-channel Transformer family | Compared Transformer-family architectures on one-channel ensemble input. |
| `CRVSE Phase B - Ensemble rPPG - CRVSETransformer family (Round 4) part 2 - multichannel (3ch).ipynb` | Multichannel Transformer family | Produced the Phase-2 app checkpoint family. Multichannel PhysFormer became the strongest Phase-2 app candidate. |
| `CRVSE MCD transfer-learning triage.ipynb` | MCD transfer triage | Checked which MCD vital-sign targets looked learnable from the frozen HR backbone. |

Main interpretation from the audit pass: the multichannel PhysFormer was not
uniformly strong across datasets, but it was the strongest practical model
candidate. It performed better on MCD-rPPG and UBFC-rPPG than on UBFC-Phys and
ECG-Fitness, which shaped the later app-scope decisions.

## Phase 2: Live-Compatible Experiments

| Notebook | Role | Conclusion |
| --- | --- | --- |
| `NB_P2_10_Live-Compatible_CRVSE_Finetune_PhysFormer.ipynb` | Shallow transfer learning | Small aggregate gains did not justify replacing the frozen source-FPS app checkpoint. |
| `NB_P2_11_Live-Compatible_PhysFormer_Training.ipynb` | Transfer, scratch, blending, interpolation | Stronger transfer improved some in-distribution metrics but hurt out-of-domain/high-HR behavior. Scratch training was not competitive. |
| `NB_P2_12_Live-Compatible_CRVSE_Optuna_Search.ipynb` | Optuna transfer search | Optuna-guided transfer improved some non-ECG metrics but did not produce an adoptable app checkpoint. |
| `NB_P2_13_App-Relevant_CRVSE_Training_Without_ECG_Fitness.ipynb` | App-scope training without ECG-Fitness | Excluded ECG-Fitness for still/seated app scope. Transfer produced modest gains, scratch remained worse overall, and no candidate passed the adoption policy. |

NB13 closed checkpoint adoption for the Phase-2 app (`app/live_hr_demo/`, now frozen):

- no NB13 candidate was adopted
- the frozen source-FPS CRVSE PhysFormer checkpoint remains the app checkpoint
- spectral consensus remains the primary app estimate
- model HR remains experimental
- ECG-Fitness is archived as a stress-test finding, not part of the current
  still/seated app claim

## Phase 3: Frame-Based Models

These notebooks moved the project from hand-crafted 1D signals to models that read the
face-crop video itself. They define the current app's input contract, its model and the
sets it is judged on. Each is listed with the question it asked and what it found.

| Notebook | Question | Result |
| --- | --- | --- |
| `NB_P3_01_RR_Exploration.ipynb` | Can respiratory rate be estimated from the facial rPPG signal with classical spectral methods, against a checkable reference? | Inconclusive (revised 2026-11-04). RIIV, RIAV and RSA correlated near zero with MCD-rPPG's clinical label (RIIV r = -0.01 at n = 504), but neither reference could tell a working method from a constant: the label is a whole-number spot value that a constant 18 breaths/min matches within 2 in 90.5% of those recordings, and the beat-timing RSA that facial RIIV was compared with came from an ECG taken at MCD's measurement stand, not recorded with the video. |
| `NB_P3_02_DLCN_Exploration.ipynb` | What does DLCN contain, and how much of it is usable? | 780 recordings of 98 people in four night-lighting scenarios, at rest and after exercise, with a frame-synced BVP. Low light does not degrade the signal, and about 99% is usable as a waveform target. |
| `NB_P3_03_DLCN_Preprocessing.ipynb` | How does DLCN go into the unified Phase-3 store? | One consolidated HDF5 with a group per recording: 72x72 face crops, the BVP label, and quality fields that weight training. |
| `NB_P3_04_MCD-rPPG_raw_dataset_exploration.ipynb` | How is the MCD-rPPG download laid out? | Three camera views and two states per subject, with ECG, PPG and a biomarker table: the file patterns the loader parses. |
| `NB_P3_05_MCD-rPPG_Preprocessing.ipynb` | How does MCD-rPPG (frontal webcam) go into the store? | The main training store: 600 subjects at rest and after exercise, with the synced PPG as BVP label and the ECG and clinical biomarkers kept. |
| `NB_P3_06_UBFC-rPPG_Preprocessing.ipynb` | How does UBFC-rPPG go into the store? | 42 resting subjects of about a minute each, the contact PPG interpolated onto the frame times as BVP. |
| `NB_P3_07_HR_DataLoader.ipynb` | How are training clips drawn from the stores? | The shared loader: a subject-wise split, fixed-length clips, temporal-rescale and flip augmentation that widens the heart-rate range above 85 bpm, and signal-quality soft weights. |
| `NB_P3_08_HR_PhysNet_Training.ipynb` | How well does PhysNet (3D CNN) learn heart rate here? First of the architecture screen. | Validation MAE 7.31 bpm (DLCN 3.24, MCD 10.69, UBFC-rPPG 2.80) at 0.77 M parameters: the screen's best for its size. |
| `NB_P3_09_HR_PhysFormer_Training.ipynb` | PhysFormer on the same pipeline? | 9.71 bpm, MCD 15.93. |
| `NB_P3_10_HR_RhythmMamba_Training.ipynb` | RhythmMamba on the same pipeline? | 6.83 bpm, the screen's lowest (MCD 10.49), at 4.92 M parameters. |
| `NB_P3_11_HR_EfficientPhys_Training.ipynb` | EfficientPhys on the same pipeline? | 12.71 bpm, MCD 20.77. |
| `NB_P3_12_HR_TYrPPG_Training.ipynb` | TYrPPG on the same pipeline? | 10.77 bpm (DLCN 7.52, MCD 13.61). |
| `NB_P3_13_HR_PhysNet_Optuna.ipynb` | Does a hyperparameter search improve PhysNet? | No: the best configuration retrained to 7.24 against the untuned 7.31 (evaluated in NB_P3_19). |
| `NB_P3_14_HR_RhythmMamba_Optuna.ipynb` | Does a hyperparameter search improve RhythmMamba? | No: the best proxy, 8.37, is worse than the untuned 6.83 (evaluated in NB_P3_19). |
| `NB_P3_15_VitalVideos_Exploration.ipynb` | What does VitalVideos-WW provide, and do video and references line up? | Per person, 60 s of 60 fps lab video and an iPhone clip, with 500 Hz PPG, ECG and respiration, skin type and blood pressure. The lab video is the model input, the heart-rate references agree, and respiration is usable behind a periodicity check. |
| `NB_P3_16_PhysDrive_Exploration.ipynb` | What does PhysDrive (drivers in vehicles) provide? | A clean frame-synced BVP under real in-vehicle motion and light. Its respiration channel is not a usable reference: cross-sensor agreement is at chance. |
| `NB_P3_17_PhysDrive_Preprocessing.ipynb` | How does PhysDrive go into the store? | The in-vehicle store, respiration masked, used in NB_P3_18's experiment. |
| `NB_P3_18_HR_PhysNet_v2.ipynb` | What do an SNR frequency loss and the in-vehicle corpus do to PhysNet? | The SNR loss is the phase's heart-rate gain: validation MAE 7.31 to 5.22, MCD 10.69 to 6.84. Adding PhysDrive repaired that corpus and made the other three worse. The run without it is the shipped `hr_physnet_v2`. |
| `NB_P3_19_Optuna_Trials_Evaluation.ipynb` | What did the two hyperparameter searches achieve? | A well-run negative: neither beat its untuned baseline, and MCD's error did not move with any setting. |
| `NB_P3_20_HR_PhysFormer_v2.ipynb` | Does the SNR loss help PhysFormer too? | Yes: 9.71 to 5.63 bpm, MCD 15.93 to 7.70. |
| `NB_P3_21_HR_RhythmMamba_v2.ipynb` | Does the SNR loss help RhythmMamba? | No: 6.83 to 7.50, DLCN 2.32 to 6.56, with unstable training. |
| `NB_P3_22_HR_RhythmFormer.ipynb` | How does RhythmFormer do with the SNR recipe? | The best in-distribution result (4.26 bpm, MCD 5.77) and the worst in-vehicle zero-shot (37.4 bpm, r 0.07). Later ruled out as the app's model: 4.3 times PhysNet's checkpoint, and too slow for capture-then-analyse. |
| `NB_P3_23_HR_PhaseNet.ipynb` | PhaseNet with its own loss or the SNR loss? | SNR 6.15 against native 10.32 (MCD 8.56 against 15.49): the SNR recipe helped four of the five architectures tried. |
| `NB_P3_24_UBFC-Phys_Exploration.ipynb` | What is in UBFC-Phys, and can it serve as a held-out set? | 56 subjects at rest, speaking and doing arithmetic, 180 s of 35 fps video with a 64 Hz wrist BVP. Adopted as held-out. |
| `NB_P3_25_UBFC-Phys_Preprocessing.ipynb` | How does UBFC-Phys go into a held-out store? | 168 recordings with the wrist BVP on the frame times, never used in training. |
| `NB_P3_26_UBFC-Phys_evaluation_review.ipynb` | When PhysNet v2 and POS agree but both miss the wrist reference (7.4% of rest windows), which is wrong? | The reference is sound and no files are swapped; timing offsets explain 3 recordings and camera failures 4. Two estimators agreeing is not corroboration. |
| `NB_P3_27_VitalVideos_WorldWide_Preprocessing.ipynb` | Are VitalVideos-WW's cohort and references sound, and how is the set split and stored? | References clean; skin type confounded with recording site; a 240/60 subject split frozen; the respiration belt confirmed in 264 of 300; the store written (34.2 GB). |
| `NB_P3_28_ECG-Fitness_Exploration.ipynb` | What does ECG-Fitness contain, before any preprocessing? | Three silent defects in the distribution and no PPG, so evaluation only; the Phase-2 halogen exclusion and camera choice reversed; an ECG reference built from three readouts that accepts 89.7% of windows. |
| `NB_P3_29_ECG-Fitness_Preprocessing.ipynb` | How does ECG-Fitness become an exercise refusal benchmark? | The benchmark protocol fixed in advance (Part 0), and 100 recordings stored behind guards that keep them out of training. |
| `NB_P3_30_HR_PhysNet_R0.ipynb` | Does the new training path reproduce v2? | Yes, narrowly: 6.45 against 5.50 bpm, inside the 1.0 bpm rule. NB_P3_31 traced the gap to bf16 training. |
| `NB_P3_31_HR_PhysNet_Full_Corpus.ipynb` | With seeds counted, do aligned labels and VitalVideos-WW in training help? | Aligned labels: no measurable effect. VitalVideos-WW's 240 training subjects: 1.39 bpm better on its 60 held-out subjects and Fitzpatrick 5-6 from 7.27 to 5.02, with the guards held, so that arm became the v3 candidate (`hr_physnet_v3`). |
| `NB_P3_32_App_Level_Battery.ipynb` | Does v3 replace v2 as the app's default, under a rule fixed before any result? | No: at Fitzpatrick 5-6 v3 reported one more recording, short of the two required. v2 stays the default; v3's gain shows as accuracy (0.70 to 0.42 bpm where both report). |
| `NB_P3_33_Rice_iPPG_Exploration.ipynb` | Can the camera time heartbeats well enough for variability, on a set recorded for it? | Camera beats follow the finger's without a slip but are timed to 17 ms at rest, so camera RMSSD runs 33% high. POS reads heart rate within 3.6 bpm in 69 of 70 recordings; the app's framing gate refuses all 70. |
| `NB_P3_34_Rice_iPPG_Preprocessing.ipynb` | How does the Rice set go into an evaluation store? | 70 recordings with both candidate crops, the reference on the frame grid and natively, and three skin-region traces; NB_P3_33 reproduced exactly. |
| `NB_P3_35_MR-NIRP_Indoor_Exploration.ipynb` | Does MR-NIRP Indoor add anything the project lacks? | Mostly a second session on Rice's own camera (7 of 8 subjects). POS is within 1.7 bpm everywhere, and the app's 8-bit crop costs about a quarter of the beat-timing error. Adopted for evaluation, not training. |
| `NB_P3_36_MR-NIRP_Indoor_Preprocessing.ipynb` | How does MR-NIRP Indoor go into an evaluation store? | 15 recordings with the app's 8-bit crop and a full-depth trace of the same region, verified byte for byte; an alignment tie found under NB_P3_35's timing figure. |
| `NB_P3_37_VitalVideos_PRV_Confirmation.ipynb` | Does v2's 60 s SDNN hold on VitalVideos-WW, under a rule fixed in advance? | No: within 10% of the contact PPG in 9.3% of 236 recordings, against a bar of 50%. "HRV" is not earned and RMSSD is closed; the Rice result stands for one laboratory camera only. |
| `NB_P3_38_HR_Robustness_Checks.ipynb` | Does the heart-rate reading hold below 60 bpm, after a browser-grade encode and at lower frame rates? Three checks, each under a rule fixed in advance. | Below 60 bpm: inconclusive (good windows 63.6% against 85.8% at 65-90 bpm; the loss is v2's, not the confidence formula's). The app's recorder setting: inconclusive (9 of 10 readings within 0.72 bpm, one marginal recording lost; POS lost 6). Frame rate: v2 holds down to 24 fps, with 22 and 20 fps unresolved. No gate moved. |
| `NB_P3_39_RR_Exploration.ipynb` | Which camera readout carries the breathing rate, against a waist belt recorded with the video? Development on VitalVideos-WW's training split, each check's method fixed before it ran. | Only the torso's motion in the whole frame: two readouts put 197 of 208 within 2 breaths/min (94.7%), against 71 for a constant at the training median. Head motion in the face crop, and the face's colour intensity, pulse amplitude and pulse-rate swing, did not beat the constant; on Rice's paced deep breathing the pulse-rate swing followed the pace while intensity and amplitude did not. Readout A (pixel components) chosen at the gate. |
| `NB_P3_40_RR_Confirmation.ipynb` | Does readout A hold on the held-out split, under a rule fixed before any of its frames was read? | Confirmed: 54 of 56 within 2 breaths/min of the belt (96.4%, 95% 87.9-99.0%), against 22 for the constant; an encode at the app's recorder setting moved no reading on 11 raw recordings. A close-up phone framing leaves too little torso below the face to read half of its recordings. The app keeps only the face crop and reports no respiratory rate. |

Main interpretation: the training loss mattered more than the architecture (NB_P3_18,
20, 21, 23), and PhysNet v2 was chosen for size and speed as well as accuracy. The later
evaluations run through the app's own code, and where a decision rests on one, its rule
was fixed before the run (NB_P3_29, 32, 37, 38 and 40).

## Current Scientific Boundary

What the notebooks support, in cautious terms; the figures and their caveats are in the
README and in Card 2 of the model card.

- **Heart rate at rest, through the app's own code.** PhysNet v2 (NB_P3_18) is the app's
  default estimator, with POS beside it as a classical cross-check that shares none of
  its assumptions. On VitalVideos-WW, which v2 never trained on, the readings the app
  marks `ok` are 0.37 bpm from the reference on average (252 recordings); UBFC-Phys at
  rest is harder (window MAE 6.97 bpm, against a wrist reference).
- **Refusal is part of the result.** The app declines a capture rather than guess when a
  gate fails: signal quality, frame rate, framing, the face checks or head motion. On
  VitalVideos-WW the skin-tone and outdoor-site disparity shows as coverage, not as wrong
  numbers; during exercise the readings follow head motion (NB_P3_28 and 29), which the
  motion gate now refuses.
- **Robustness, under rules fixed in advance (NB_P3_38).** Below 60 bpm v2 keeps fewer
  usable windows (inconclusive against the bar); an encode at the app's recorder setting
  kept 9 of 10 readings within 0.72 bpm (inconclusive); on lossless crops thinned to lower
  rates the reading holds down to 24 fps, below the 27 fps flag, with 22 and 20 fps
  unresolved. No gate was moved.
- **Measured negatives:** pulse-rate variability, recoverable as 60 s SDNN on one
  laboratory camera (NB_P3_33) but not confirmed on an independent set under a rule fixed
  in advance (NB_P3_37), with RMSSD not recoverable; in-vehicle video (NB_P3_18);
  hyperparameter tuning (NB_P3_19).
- **Respiratory rate, on lab video and from the torso (NB_P3_40).** Under a rule fixed
  before any held-out frame was read, the motion of the torso below the face, in the whole
  frame, read the breathing rate within 2 breaths/min of a waist belt in 54 of 56 held-out
  VitalVideos-WW recordings, and an encode at the app's recorder setting moved no reading.
  The face crop's own signals did not beat a constant (NB_P3_39), and the app keeps only
  the crop, so it reports no respiratory rate. NB_P3_01's classical test, with no
  reference recorded with the video, stays inconclusive.
- **A candidate, not a claim:** `hr_physnet_v3` (NB_P3_31) is registered beside v2; the
  app-level battery (NB_P3_32) kept v2 as the default.
- **Not characterised:** elevated heart rate and iOS devices.

The app is a research demo: not a medical device, not diagnostic and not clinically
validated.