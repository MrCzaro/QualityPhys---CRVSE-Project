# CRVSE Notebook Index

This file summarizes the main notebook chronology and conclusions for the
QualityPhys / CRVSE project. It is a navigation and interpretation aid, not a
claim that every notebook was freshly rerun during this documentation pass.

## Reading Guide

The notebooks fall into four groups:

- dataset preprocessing
- POS-only model experiments
- ensemble POS/CHROM/GREEN model experiments
- live-compatible checkpoint adaptation and app-scope experiments

The later notebooks and audit reports should be trusted more than early
exploratory notebooks when they disagree, because they were written after the
live preprocessing and app-scope issues became clearer.

## Dataset Preprocessing

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

## POS-Only Model Experiments

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
triage, but they were not the final app contract. The live app and current model
use multichannel POS/CHROM/GREEN inputs.

## Ensemble And Multichannel Model Experiments

These notebooks trained model families on rPPG ensemble artifacts. They are the
main source of the current app checkpoint.

| Notebook | Role | Notes |
| --- | --- | --- |
| `CRVSE Phase B - Ensemble rPPG - CRVSENET (Round 1).ipynb` | Ensemble model baseline | Tested CRVSENet-style learning on ensemble rPPG. |
| `CRVSE Phase B - Ensemble rPPG - CRVSEInceptionNet 1D (Round 2) part 1 - ensemble only (1D).ipynb` | One-channel Inception | Tested InceptionNet on one-channel ensemble rPPG. |
| `CRVSE Phase B - Ensemble rPPG - CRVSEInceptionNet 1D (Round 2) part 2 - multichannel (3ch).ipynb` | Multichannel Inception | Tested InceptionNet on POS/CHROM/GREEN. |
| `CRVSE Phase B - Ensemble rPPG - CRVSEResNet 1D (Round 3) part 1 - ensemble only (1ch).ipynb` | One-channel ResNet | Tested ResNet on one-channel ensemble rPPG. |
| `CRVSE Phase B - Ensemble rPPG - CRVSEResNet 1D (Round 3) part 2 - multichannel (3ch).ipynb` | Multichannel ResNet | Tested ResNet on POS/CHROM/GREEN. |
| `CRVSE Phase B - Ensemble rPPG - CRVSETransformer family (Round 4) part 1 - ensemble only (1ch).ipynb` | One-channel Transformer family | Compared Transformer-family architectures on one-channel ensemble input. |
| `CRVSE Phase B - Ensemble rPPG - CRVSETransformer family (Round 4) part 2 - multichannel (3ch).ipynb` | Multichannel Transformer family | Produced the current app checkpoint family. Multichannel PhysFormer became the strongest app candidate. |
| `CRVSE MCD transfer-learning triage.ipynb` | MCD transfer triage | Checked which MCD vital-sign targets looked learnable from the frozen HR backbone. |

Main interpretation from the audit pass: the multichannel PhysFormer was not
uniformly strong across datasets, but it was the strongest practical model
candidate. It performed better on MCD-rPPG and UBFC-rPPG than on UBFC-Phys and
ECG-Fitness, which shaped the later app-scope decisions.

## Live-Compatible Experiments

| Notebook | Role | Conclusion |
| --- | --- | --- |
| `NB_P2_10_Live-Compatible_CRVSE_Finetune_PhysFormer.ipynb` | Shallow transfer learning | Small aggregate gains did not justify replacing the frozen source-FPS app checkpoint. |
| `NB_P2_11_Live-Compatible_PhysFormer_Training.ipynb` | Transfer, scratch, blending, interpolation | Stronger transfer improved some in-distribution metrics but hurt out-of-domain/high-HR behavior. Scratch training was not competitive. |
| `NB_P2_12_Live-Compatible_CRVSE_Optuna_Search.ipynb` | Optuna transfer search | Optuna-guided transfer improved some non-ECG metrics but did not produce an adoptable app checkpoint. |
| `NB_P2_13_App-Relevant_CRVSE_Training_Without_ECG_Fitness.ipynb` | App-scope training without ECG-Fitness | Excluded ECG-Fitness for still/seated app scope. Transfer produced modest gains, scratch remained worse overall, and no candidate passed the adoption policy. |

NB13 is the current project conclusion for checkpoint adoption:

- no NB13 candidate was adopted
- the frozen source-FPS CRVSE PhysFormer checkpoint remains the app checkpoint
- spectral consensus remains the primary app estimate
- model HR remains experimental
- ECG-Fitness is archived as a stress-test finding, not part of the current
  still/seated app claim


## Current Scientific Boundary

The notebooks support a cautious portfolio conclusion:

- the model has offline value on the preprocessed training distribution
- live inference is more fragile because acquisition, FPS, lighting, movement,
  and preprocessing can differ from training
- ECG-Fitness remains a useful stress-test domain but is not supported by the
  current live app claim
- the app should continue to show spectral HR as primary and model HR as
  experimental
