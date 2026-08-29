# Third-Party Attribution and Citations

Attribution for the model architectures, reference implementations, and datasets used
in the CRVSE research arc. Datasets are covered separately in `docs/data_sources.md`;
this file covers code and published architectures.

## Redistribution status

No third-party source code is contained in this repository. The Phase-3 architecture
notebooks clone their reference implementations at runtime (`git clone --depth 1`) on
the training host, so nothing is redistributed here and no bundled-NOTICE obligation
arises from them. `app/live_vitals/models/architectures/physnet.py` and
`app/live_hr_demo/models/architectures/crvse_physformer.py` are original
implementations of published architectures, not copies of the authors' code.

Citation obligations still apply, and two licence points below need attention before
any distribution or commercial use.

## Model architectures

| Architecture | Paper | Reference implementation | Licence | Used in |
|---|---|---|---|---|
| PhysNet | Yu et al., BMVC 2019 | not cloned — reimplemented | n/a (own code) | NB_P3_08, NB_P3_18, live_vitals app |
| PhysFormer | Yu et al., CVPR 2022 | `ZitongYu/PhysFormer` | MIT | NB_P3_09, NB_P3_20 |
| RhythmMamba | Zou et al., arXiv 2024 | `zizheng-guo/RhythmMamba` | MIT | NB_P3_10, NB_P3_21 |
| RhythmFormer | Zou et al., Pattern Recognition 2025 | `zizheng-guo/RhythmFormer` | MIT | NB_P3_22 |
| EfficientPhys | Liu et al., WACV 2023 | `ubicomplab/rPPG-Toolbox` | **RAIL v1.1** | NB_P3_11 |
| TYrPPG | Chen et al., IEEE WI-IAT 2025 (AI4SG workshop) | `Taixi-CHEN/TYrPPG` | MIT | NB_P3_12 |
| PhaseNet | Zhao et al., CVPR 2026 | `Alex036225/PhaseNet` | **none stated** | NB_P3_23 |

`ubicomplab/rPPG-Toolbox` is additionally the general code reference for the Phase-3
modelling approach, and the PhaseNet authors ask that it be cited alongside their work.

## Licence points requiring attention

**rPPG-Toolbox is under a Responsible AI Source Code License (RAIL v1.1), not a
permissive licence.** It permits commercial use but imposes behavioural use
restrictions, including a healthcare category that prohibits diagnosing medical
conditions without human oversight, and restrictions covering surveillance, inference
of protected characteristics, synthetic media, and criminal-risk prediction. CRVSE's
stated scope — a research and portfolio demo, explicitly not a medical device, not
diagnostic, and not for clinical decision-making — is consistent with these terms, and
that scope should be preserved in any derived work. Review the full licence before any
distribution or commercial use.

**PhaseNet states no licence.** Absent an explicit grant, default copyright reserves
all rights. It was cloned at runtime for a single research comparison (NB_P3_23, a
documented negative result) and none of its code is redistributed here. Contact the
authors before reusing or redistributing that code.

**Reproducibility caveat.** The notebooks clone from each repository's default branch
without pinning a commit, so upstream changes can silently break reproduction of
NB_P3_09 through NB_P3_23. Pinning a commit SHA per clone would resolve this.

## Citations

```bibtex
@inproceedings{yu2019remote,
  title={Remote Photoplethysmograph Signal Measurement from Facial Videos Using
         Spatio-Temporal Networks},
  author={Yu, Zitong and Li, Xiaobai and Zhao, Guoying},
  booktitle={Proc. British Machine Vision Conference (BMVC)},
  year={2019}
}

@inproceedings{yu2021physformer,
  title={PhysFormer: Facial Video-based Physiological Measurement with Temporal
         Difference Transformer},
  author={Yu, Zitong and Shen, Yuming and Shi, Jingang and Zhao, Hengshuang and
          Torr, Philip and Zhao, Guoying},
  booktitle={CVPR},
  year={2022}
}

@article{zou2024rhythmmamba,
  title={RhythmMamba: Fast remote physiological measurement with arbitrary length
         videos},
  author={Zou, Bochao and Guo, Zizheng and Hu, Xiaocheng and Ma, Huimin},
  journal={arXiv preprint arXiv:2404.06483},
  year={2024}
}

@article{zou2025rhythmformer,
  title={RhythmFormer: Extracting patterned rPPG signals based on periodic sparse
         attention},
  author={Zou, Bochao and Guo, Zizheng and Chen, Jiansheng and Zhuo, Junbao and
          Huang, Weiran and Ma, Huimin},
  journal={Pattern Recognition},
  volume={164},
  pages={111511},
  year={2025}
}

@inproceedings{liu2023efficientphys,
  title={EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Cardiac
         Measurement},
  author={Liu, Xin and Hill, Brian and Jiang, Ziheng and Patel, Shwetak and
          McDuff, Daniel},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of
             Computer Vision (WACV)},
  pages={5008--5017},
  year={2023}
}

@article{liu2022rppgtoolbox,
  title={rPPG-Toolbox: Deep Remote PPG Toolbox},
  author={Liu, Xin and Narayanswamy, Girish and Paruchuri, Akshay and Zhang, Xiaoyu
          and Tang, Jiankai and Zhang, Yuzhe and Wang, Yuntao and Sengupta,
          Soumyadip and Patel, Shwetak and McDuff, Daniel},
  journal={arXiv preprint arXiv:2210.00716},
  year={2022}
}

@inproceedings{zhao2026phasenet,
  title={Phase-Net: Physics-grounded harmonic attention system for efficient remote
         photoplethysmography measurement},
  author={Zhao, Bo and Guo, Dan and Cao, Junzhe and Xu, Yong and Zou, Bochao and
          Tan, Tao and Sun, Yue and Yu, Zitong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
             Recognition (CVPR)},
  pages={21198--21207},
  year={2026}
}
```

TYrPPG is published as an IEEE WI-IAT 2025 AI4SG workshop paper (arXiv:2511.05833);
its repository does not provide a BibTeX entry.

## Supporting libraries

MediaPipe Face Landmarker, PyTorch, OpenCV, NumPy, SciPy, FastHTML and MonsterUI are
ordinary runtime dependencies, installed rather than redistributed, and carry no
bundled-NOTICE obligation here. The MediaPipe Face Landmarker model asset
(`face_landmarker.task`) is a downloaded Google model file kept out of version control
and governed by its own terms. `einops`, `timm`, `thop`, `mamba-ssm` and
`causal-conv1d` are notebook-only training dependencies.

## Datasets

Dataset provenance, licensing boundaries and redistribution limits are documented in
`docs/data_sources.md`. Raw datasets are external materials; this repository's
Apache-2.0 licence does not grant rights to redistribute them.
