# P2-MCA-SA

Official reproducibility package for the manuscript **“Cross-scale trade-offs and assignment diagnosis of the P2 detection head in UAV small-object detection.”**

The repository studies three components in a controlled YOLO setting:

- a stride-4 P2 detection head;
- multi-branch cross attention (MCA) applied to the P2 path; and
- scale-aware task-aligned assignment (SA), implemented as a training-time eligibility constraint.

The main claim is deliberately bounded. On the three-seed 1280-pixel protocol, the full P2+MCA+SA configuration improves the point estimates of overall AP and small-object AP by 0.67 and 0.62 percentage points over the YOLOv8m baseline. None of the ten prespecified paired comparisons remains significant at 0.05 after Holm correction. The repository therefore supports diagnosis of scale-dependent behaviour rather than a universal claim that adding P2 is always beneficial.

## Repository contents

```text
P2-MCA-SA/
├── configs/
│   ├── data/                  # Three-class and official 10-class data YAMLs
│   └── models/                # P2, MCA, SA-compatible, and transfer configs
├── nmv/
│   ├── modules/               # MCA and auxiliary model modules
│   ├── patches/               # SA, TAL audit, registration, and runtime patches
│   └── utils/
├── scripts/
│   ├── build_nmv3_from_visdrone.py
│   ├── train.py
│   ├── run_p2_sa_ablation.py
│   ├── eval_size_buckets.py
│   └── ...
├── splits/                    # Exact image-stem manifests (4555/250/251)
├── results/                   # Derived tables and TAL diagnostic summaries
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

The repository does **not** redistribute VisDrone images or annotations, trained weights, private manuscript drafts, or machine-specific experiment directories.

## Installation

Python 3.9 or newer is recommended. Install a PyTorch build compatible with your CUDA version first, then install the remaining dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/sanity_check.py
```

The code was developed against `ultralytics==8.4.37`. GPU memory, batch size, and runtime will vary by hardware.

## Dataset preparation

Download the official VisDrone2019-DET train and validation packages from the [VisDrone dataset repository](https://github.com/VisDrone/VisDrone-Dataset). The public files should contain `images/` and `annotations/` directories.

Rebuild the exact NMV-SOD-3cls split used by the manuscript:

```bash
python scripts/build_nmv3_from_visdrone.py \
  --train-root /path/to/VisDrone2019-DET-train \
  --val-root /path/to/VisDrone2019-DET-val \
  --output datasets/nmv_visdrone_3cls
```

The conversion uses the following official-category mapping:

| VisDrone category | Repository class |
|---|---|
| motor (10) | 0: `ebike` |
| bicycle (3) | 1: `bicycle` |
| tricycle (7), awning-tricycle (8) | 2: `etrike` |

The `splits/` manifests define the exact 4555 training, 250 validation, and 251 test images. The validation and test partitions are disjoint subsets of the official VisDrone validation package.

Set the dataset location before training:

```bash
# Linux/macOS
export NMV_DATA_ROOT=/absolute/path/to/datasets/nmv_visdrone_3cls

# Windows PowerShell
$env:NMV_DATA_ROOT = "D:/datasets/nmv_visdrone_3cls"
```

For the official ten-class calibration, convert VisDrone with `scripts/build_visdrone10.py` and set `VISDRONE10_DATA_ROOT`.

## Core reproduction protocol

All formal three-class comparisons use 150 epochs and three matched seeds (`42`, `1`, and `7`). The canonical headline protocol uses 1280-pixel input.

Baseline and P2:

```bash
python scripts/train.py --only 1   # YOLOv8m baseline
python scripts/train.py --only 2   # +P2
```

P2+SA isolated ablation:

```bash
python scripts/run_p2_sa_ablation.py --stage confirm
```

Full P2+MCA+SA configuration (`idx=17`):

```bash
# seed 42
NMV_RUN_SUFFIX=_1280 NMV_SEED=42 python scripts/train.py --only 17

# matched additional seeds
NMV_RUN_SUFFIX=_1280_s1 NMV_SEED=1 python scripts/train.py --only 17
NMV_RUN_SUFFIX=_1280_s7 NMV_SEED=7 python scripts/train.py --only 17
```

On PowerShell, set the same values with `$env:NAME = "value"` before invoking Python. Experiment `17` enables SA with `NMV_SCALE_HI_RATIO=32` internally and uses the MCA model configuration.

Evaluate standard and COCO size-bucket metrics:

```bash
python scripts/val.py --run E16_mca_scale_assign_hi32_1280 --split test --imgsz 1280
python scripts/eval_size_buckets.py --run E16_mca_scale_assign_hi32_1280 --split test --imgsz 1280
```

The TAL audit can be enabled with `NMV_TAL_AUDIT=1`; `scripts/audit_tal_checkpoint.py` and `scripts/scale_assign_eligibility.py` reproduce the eligibility and post-top-k diagnostics.

## Results and provenance

`results/canonical_3seed.csv` contains the per-seed values used for the final four-configuration comparison. `results/canonical_3seed.json` additionally stores means, sample standard deviations, paired tests, and Holm-adjusted p-values. The diagnostic JSON files record P2 and P2+SA TAL assignment summaries at seed 42.

Raw datasets, full prediction archives, TensorBoard logs, and model checkpoints are excluded because of licensing and size. The included manifests, configurations, scripts, and derived result tables are sufficient to reconstruct the documented protocol after obtaining VisDrone.

## License

This project integrates with the AGPL-3.0 distribution of Ultralytics YOLO. The repository is released under the GNU Affero General Public License v3.0; see [LICENSE](LICENSE).

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). Until a DOI is assigned, cite this repository together with the manuscript title above.

