"""Main training entry for the non-motor-vehicle small-target detection paper.

Each experiment is dispatched as an isolated subprocess so monkey-patches
(MPDIoU loss / Soft-NMS) cannot leak between experiments. The parent reads
results.csv after each child returns and prints a summary table at the end.

Experiments E01-E11 (unified condition: imgsz=960, epochs=150, patience=50, batch=4, seed=42).
- E01-E05: EMA-branch ablation chain (baseline → +P2 → +EMA → +MPDIoU → +GFPN)
- E06-E07: inference enhancements (Soft-NMS / TTA on E05 best.pt)
- E08-E11: ⭐ Two new innovations of this work (MCA + CAGFPN)
- SAHI inference handled separately by scripts/sahi_predict.py

Usage in VSCode terminal:
  cd P2-MCA-SA
  python scripts/train.py                 # bare run = the paper 1280 main line (PAPER_QUEUE):
                                          #   trains baseline/+P2/+P2+MCA plus 2 extra seeds,
                                          #   each evaluated on test as it finishes; everything written to new _1280
                                          #   directories, leaving the old 960 results alone; Ctrl+C any time, re-running resumes.
  python scripts/train.py --only 8        # (advanced) run a single EXPERIMENTS entry (old 960 naming)
  python scripts/train.py --start 8       # (advanced) run the old EXPERIMENTS list from a given idx
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nmv  # triggers patch injection (env-var driven)
from nmv.utils.metrics_io import is_completed, read_best
from nmv.utils.data import ensure_data_yaml


# NMV_DATA overrides the dataset yaml (e.g. configs/data/visdrone10.yaml for the ten-class anchor experiment); defaults to the three-class main dataset
DATA = ensure_data_yaml(Path(os.environ.get(
    "NMV_DATA", str(ROOT / "configs" / "data" / "nmv_visdrone_3cls.yaml"))))
MODELS = ROOT / "configs" / "models"
RUNS = ROOT / "runs"
WEIGHTS = ROOT / "weights"


EXPERIMENTS = [
    # ========= EMA branch, cumulative ablation E01-E05 (comparison baselines) =========
    dict(
        idx=1, name="E01_baseline",
        kind="train", cfg=None, weights="yolov8m.pt",
        env={},
        desc="YOLOv8m baseline (3 detection heads, CIoU, hard NMS)",
    ),
    dict(
        idx=2, name="E02_p2",
        kind="train", cfg=str(MODELS / "yolov8m-p2.yaml"), weights="yolov8m.pt",
        env={},
        desc="+P2 detection head (4 scales: P2/4, P3/8, P4/16, P5/32)",
    ),
    dict(
        idx=3, name="E03_p2_ema",
        kind="train", cfg=str(MODELS / "yolov8m-p2-ema.yaml"), weights="yolov8m.pt",
        env={},
        desc="+EMA attention (factor=4) on P2 features",
    ),
    dict(
        idx=4, name="E04_p2_ema_mpdiou",
        kind="train", cfg=str(MODELS / "yolov8m-p2-ema.yaml"), weights="yolov8m.pt",
        env={"NMV_IOU": "mpdiou"},
        desc="+MPDIoU regression loss (Ma, arXiv:2307.07662)",
    ),
    dict(
        idx=5, name="E05_p2_ema_mpdiou_gfpn",
        kind="train", cfg=str(MODELS / "yolov8m-p2-ema-gfpn.yaml"), weights="yolov8m.pt",
        env={"NMV_IOU": "mpdiou"},
        desc="EMA-branch full: P2 + EMA + MPDIoU + GFPN cross-scale skip",
    ),

    # ========= E05 inference-time enhancements E06-E07 (val_only, reusing the E05 best.pt) =========
    dict(
        idx=6, name="E06_softnms",
        kind="val_only", source="E05_p2_ema_mpdiou_gfpn",
        env={"NMV_IOU": "mpdiou", "NMV_NMS": "soft"},
        desc="E05 best.pt + Soft-NMS at inference (no retrain)",
    ),
    dict(
        idx=7, name="E07_tta",
        kind="val_only", source="E05_p2_ema_mpdiou_gfpn",
        env={"NMV_IOU": "mpdiou"}, val_kwargs=dict(augment=True),
        desc="E05 best.pt + Test-Time Augmentation (multi-scale + flip)",
    ),
    # (Soft-NMS + TTA SAHI is run separately from scripts/sahi_predict.py and is not an EXPERIMENTS entry)

    # ========= the two components of this work, E08-E11 (MCA + CAGFPN) =========
    dict(
        idx=8, name="E08_p2_mca",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca.yaml"), weights="yolov8m.pt",
        env={},
        desc="+MCA multi-branch cross attention (this work: EMA x CoordAtt gated fusion)",
    ),
    dict(
        idx=9, name="E09_p2_mca_mpdiou",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca.yaml"), weights="yolov8m.pt",
        env={"NMV_IOU": "mpdiou"},
        desc="+MPDIoU (on MCA base)",
    ),
    dict(
        idx=10, name="E10_p2_mca_mpdiou_gfpn",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-gfpn.yaml"), weights="yolov8m.pt",
        env={"NMV_IOU": "mpdiou"},
        desc="+GFPN (MCA base, no context augmentation)",
    ),
    dict(
        idx=11, name="E11_full_mca_cagfpn",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-cagfpn.yaml"), weights="yolov8m.pt",
        env={"NMV_IOU": "mpdiou"},
        desc="Full model: P2 + MCA + MPDIoU + CAGFPN (both components of this work)",
    ),

    # ========= external comparison baselines E12-E13 (other YOLOv8 sizes) =========
    # Used for the size-vs-accuracy Pareto curve in the paper, to place E11 on the efficiency/accuracy trade-off
    # Note: if yolov8s.pt / yolov8l.pt are absent from weights/, ultralytics downloads them automatically
    # yolov8l may need NMV_BATCH=2 to fit imgsz=960 on an 8 GB GPU
    dict(
        idx=12, name="E12_yolov8s_baseline",
        kind="train", cfg=None, weights="yolov8s.pt",
        env={},
        desc="YOLOv8s baseline (smaller backbone for size-accuracy Pareto curve)",
    ),
    dict(
        idx=13, name="E13_yolov8l_baseline",
        kind="train", cfg=None, weights="yolov8l.pt",
        env={},
        desc="YOLOv8l baseline (larger backbone for size-accuracy Pareto curve)",
    ),

    # ========= MCA isolation ablation E14 (mAP_large repair experiment) =========
    # Diagnosis 2026-05-19: the mAP_large collapse in E08/E11 comes from the MCA output being
    # downsampled along the bottom-up path and contaminating the large-object features of P3/P4/P5.
    # E14 changes only line 20 of the yaml: the bottom-up downsample is sourced from the pre-MCA P2
    # (idx=18) instead of the MCA output (idx=-1=19), localising MCA to the P2 detection head.
    dict(
        idx=14, name="E14_full_mca_cagfpn_isolated",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-cagfpn-isolated.yaml"),
        weights="yolov8m.pt",
        env={"NMV_IOU": "mpdiou"},
        desc="⭐ MCA-isolated variant of E11: BU sources from pre-MCA P2 (recover mAP_large)",
    ),

    # ========= R2 defensive counter-experiment E15 (symmetric MCA, expected to be worse) =========
    # Designed 2026-05-19: adding MCA to the P5 head as well, on top of E14, to answer the anticipated
    # reviewer question "why not simply add an MCA on P5 too" - symmetric MCA should hurt, because MCA
    # is designed for dense small objects and on P5 it over-suppresses the off-centre activations that
    # large-object boundary regression needs.
    # Expectation: AP_large degrades further, +~1.5M parameters, overall mAP does not rise.
    dict(
        idx=15, name="E15_full_mca_cagfpn_p5mca",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-cagfpn-p5mca.yaml"),
        weights="yolov8m.pt",
        env={"NMV_IOU": "mpdiou"},
        desc="R2-defense: symmetric MCA on P5 (negative-result ablation for MCA-L)",
    ),

    # ========= large-object repair E16 (scale-aware label assignment) =========
    # Diagnosis reversed 2026-05-22: the root cause of the mAP_large collapse is the P2 head itself
    # (E01 0.509 -> E02 0.303). Large objects are assigned to high-resolution P2 anchors that cannot
    # regress them, starving the P5 head. E16 = E08 (P2+MCA) + size-range assignment (upper bound only:
    # each head is only assigned targets with max_side < hi_ratio*stride). A clean control against E08,
    # isolating the effect of the assignment change on large objects. Enabled by NMV_SCALE_ASSIGN=1.
    dict(
        idx=16, name="E16_mca_scale_assign",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1"},
        desc="P1: P2+MCA + scale-aware label assignment (recover mAP_large)",
    ),

    # ========= E16 HI_RATIO=32 ablation (added 2026-05-28) =========
    # E16 (HI_RATIO=16) gave test bucket mAP_large=0.248, 5.3 pp *below* E08 (P2+MCA, large=0.301),
    # falsifying the idea that an upper-bound assignment alone repairs large objects. E16 hi32 widens
    # the P4 acceptance range to max_side < 32x16 = 512 px so that medium-to-large objects can also take
    # positives on P4, testing whether HI_RATIO is the decisive factor.
    dict(
        idx=17, name="E16_mca_scale_assign_hi32",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32"},
        desc="P1+: P2+MCA + scale-aware (HI_RATIO=32, P4 widened to <=512px GT)",
    ),

    # ========= E17 P5 Transformer global branch (added 2026-06-02) =========
    # E16' (HI=32) lifted mAP_large from 0.247 to 0.390 but still short of the 0.509 baseline.
    # The residual -11.9 pp is attributed to the quality of the P5 features themselves: a stride-32
    # CNN-only path has too small a receptive field / too little global context on large objects.
    # E17 adds a 2-layer Transformer encoder (dim=192, 4 heads) on P5 (30x30 tokens at imgsz=960),
    # gated-residual fused with the CNN P5 (proj_out zero-initialised, so it starts as the identity).
    # Trained together with scale-aware (HI=32).
    dict(
        idx=18, name="E17_p2_mca_p5trans_scale",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-p5trans.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32"},
        desc="P2 + MCA + P5 Transformer + scale-aware (E17)",
    ),

    # ========= E18 external SOTA comparison baseline, YOLOv11m (added 2026-06-02) =========
    # Reviewers will ask why there is no comparison against a recent SOTA model. This entry trains
    # the stock ultralytics yolo11m.pt (downloaded automatically on first use) on the same dataset at
    # imgsz=960 batch=4, fully aligned with E01-E17, to produce a fair comparison. About 25 h of GPU time.
    dict(
        idx=19, name="E18_yolov11m_baseline",
        kind="train", cfg=None, weights="yolo11m.pt",
        env={},
        desc="External SOTA baseline: YOLOv11m on NMV-SOD-3cls (paper Tab.6 SOTA cmp)",
    ),

    # ========= E17 ablation suite (added 2026-06-02, for the ablation section) =========
    # The E17 configuration (dim=192, depth=2) is already the smallest stable one. The ablation sweeps:
    #   (a) dim ∈ {128, 192, 256} at depth=2 → E20 / E17 / E21
    #   (b) depth ∈ {1, 2, 4} at dim=192    → E22 / E17 / E23
    # 4 new configurations sharing E17 as the centre. The gate ablation runs on E17 through the
    # NMV_P5TRANS_MODE environment variable (full / trans_only / cnn_only) and needs no new yaml.
    dict(
        idx=20, name="E20_p5trans_d128_L2",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-p5trans-d128-L2.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32"},
        desc="ablation: P5 Transformer dim=128 depth=2 (smaller dim variant)",
    ),
    dict(
        idx=21, name="E21_p5trans_d256_L2",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-p5trans-d256-L2.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32"},
        desc="ablation: P5 Transformer dim=256 depth=2 (larger dim variant)",
    ),
    dict(
        idx=22, name="E22_p5trans_d192_L1",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-p5trans-d192-L1.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32"},
        desc="ablation: P5 Transformer dim=192 depth=1 (shallower variant)",
    ),
    dict(
        idx=23, name="E23_p5trans_d192_L4",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-p5trans-d192-L4.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32"},
        desc="ablation: P5 Transformer dim=192 depth=4 (deeper variant)",
    ),

    # ========= E24 transfer check (added 2026-06-18): port the components onto the YOLOv11m backbone =========
    # The plain YOLOv11m baseline E18 beats the v8-based configuration E16' on both overall and small-object
    # mAP, while being smaller. The reason is that this compares "old backbone (v8m) + hand-built modules"
    # against "new backbone (v11m)", which is not like for like. E24 = YOLOv11m + P2 head + MCA + scale-aware
    # assignment (HI=32), compared against E18 under matched conditions. The cfg backbone is copied from the
    dict(
        idx=24, name="E24_yolo11m_p2_mca_scale",
        kind="train", cfg=str(MODELS / "yolo11m-p2-mca.yaml"), weights="yolo11m.pt",
        # NMV_BATCH=1: with v11m (width 1.0) + the high-resolution P2 head, a batch-2 forward at 1280 fills
        #   7.18 GB, leaving no GPU headroom for label assignment (TaskAlignedAssigner; mosaic multi-box x the
        #   huge P2 anchor count makes the align_metric matrix enormous), so it keeps falling back to the CPU
        #   (GPU idles, system RAM fills). An 8 GB card can only do batch=1 (peak ~4.3 GB, TAL stays on GPU).
        #   Cost: BN is weaker on single samples, though nbs=64 gradient accumulation preserves the optimisation.
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32", "NMV_GPU_LIMIT_GB": "7.5", "NMV_BATCH": "1"},
        desc="transfer check: YOLOv11m + P2 + MCA + scale-aware(HI=32) vs E18 (8GB -> batch1)",
    ),

    # ========= E25 matched v11 baseline (added 2026-06-21): the batch=1 version of E18 =========
    # E24 can only run at batch=1 on 8 GB, while the original E18 (idx 19) is batch=2, so comparing them
    # across batch sizes is unfair (single-sample BN plus a different optimisation trajectory - a reviewer
    # would rightly object). E25 = plain YOLOv11m at batch=1 under exactly E24's conditions (imgsz=1280 /
    # SGD / 150 ep / same augmentation), removing the batch confound so the comparison is apples to apples.
    # cfg=None + weights=yolo11m.pt loads and fine-tunes the full COCO-pretrained yolo11m (as in E18, batch aside).
    # Does not overwrite the original E18 (batch=2) results (the run dir carries a _b1 suffix).
    dict(
        idx=25, name="E18_yolov11m_baseline_b1",
        kind="train", cfg=None, weights="yolo11m.pt",
        env={"NMV_BATCH": "1", "NMV_GPU_LIMIT_GB": "7.5"},
        desc="matched v11 baseline: YOLOv11m @batch=1 (apples-to-apples vs E24; removes the batch confound)",
    ),

    # ========= key additional ablation E26: P2 + SA (without MCA) =========
    # Isolates the contribution of scale-aware assignment itself, completing the five-row ablation
    # baseline / P2 / P2+SA / P2+MCA / P2+MCA+SA. HI_RATIO=32 matches the final model E16'.
    dict(
        idx=26, name="E26_p2_scale_assign_hi32",
        kind="train", cfg=str(MODELS / "yolov8m-p2.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32"},
        desc="key additional ablation: P2 + scale-aware assignment (HI_RATIO=32, no MCA)",
    ),
]


HP = dict(
    epochs=150,
    imgsz=1280,   # 960->1280 for the paper. Native images go up to 1920x1080 (median 1360x765), so 960 downsamples;
                  #           1280 is the single largest small-object gain. The old 960 results are kept (different run suffix) as a resolution ablation.
    batch=2,      # 4->2: activations at imgsz 1280 grow by (1280/960)^2 = 1.78x, so batch=4 always OOMs on 8 GB; batch=2 peaks at ~6.7 GB.
                  #      NMV_BATCH=3 can be tried (smoke test first); small-batch BatchNorm noise is absorbed by mosaic/mixup + warmup, no SyncBN needed.
    patience=50,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.3,
    close_mosaic=15,
    workers=0,    # 2->0: on Windows each DataLoader subprocess imports torch and takes ~1 GB of virtual memory; over a long run the address-space fragmentation was the main cause of the mid-Phase-A OOM. Retries confirmed workers=0 runs; cache='disk' offsets the IO cost
    cache='disk',
    amp=True,
    seed=42,
    device=0,
    verbose=True,
    plots=True,
)


MAX_ATTEMPTS = 3   # consecutive "zero epoch progress" retries allowed; only exhausting them (a real hang) calls sys.exit(1). Any new epoch written to disk resets it (see the progress-aware retry in dispatch_one)


# 2026-05-28: experiments skipped by default on a bare run of train.py. The reason for each is in the comment below.
# They stay in the EXPERIMENTS table for provenance. To run one of them on its own,
# use `--only <idx>` (--only bypasses DEFAULT_SKIP and does what the user asked).
DEFAULT_SKIP_REASONS = {
    6:  "Soft-NMS patch broken: mAP 0.497→0.206, postprocess 200× (see project_soft_nms_broken)",
    7:  "TTA val_only: the paper already has these numbers, and every bare run would redo ~10 min of work",
    12: "YOLOv8s baseline: external comparison baseline, not needed right now (enable when the paper calls for it)",
    13: "YOLOv8l baseline: as above, and needs batch=2 on an 8 GB card",
    15: "P5 MCA: training stopped by hand at epoch 32 on 2026-05-22; the CSAC/MCA-localisation line was dropped",
}
DEFAULT_SKIP = list(DEFAULT_SKIP_REASONS.keys())

EXPERIMENTS_BY_IDX = {e["idx"]: e for e in EXPERIMENTS}

# The paper main-line queue. A bare `python scripts/train.py` trains in this order (imgsz/batch come from HP=1280/2 above),
# evaluating each on test as it finishes. Everything carries a _1280[_sN] run suffix, so it lands in new directories and does not overwrite the old 960 results.
# (idx, seed, suffix). Most important first; Ctrl+C any time, re-running resumes.
PAPER_QUEUE = [
    # ===== two-backbone main line (reordered 2026-06-21) =====
    # Main line = scale-aware LA + MCA is backbone-agnostic: v8 (E01 -> E16', already 3-seed) + v11 (E18 -> E24).
    # One bare `python scripts/train.py` will: (1) skip completed entries instantly (test/buckets already on disk) ->
    # (2) resume E24 to 150 epochs (auto test + bucket eval) -> (3) train the E18-batch1 matched baseline (auto test + bucket eval).
    # Ctrl+C at any point; running it again resumes.
    #
    # --- completed (training + test + size-buckets all on disk): skipped instantly, auto_eval_test is idempotent; kept for provenance ---
    (17, 42, "_1280"),      #   E16' v8 main configuration, seed 42
    (17, 1,  "_1280_s1"),   #   E16' v8 hero seed1
    (17, 7,  "_1280_s7"),   #   E16' v8 hero seed7
    (2,  42, "_1280"),      #   +P2, middle row of the 1280 diagnostic chain
    (1,  42, "_1280"),      #   baseline, anchor of the 1280 diagnostic chain
    (19, 42, "_1280"),      #   E18 v11m baseline (batch=2; kept as a batch-sensitivity reference)
    # --- still to train ---
    (24, 42, "_1280"),      # RESUME E24 from 108 to 150 epochs, so it matches the 150 ep of E25. best.pt currently sits at ep99 on a plateau; after topping up, check whether best changed, and if it did, delete eval_test[_buckets] and re-evaluate. YOLOv11m+P2+MCA+scale-aware(HI=32) batch=1
    (25, 42, "_1280"),      # E18 batch=1 matched baseline (apples-to-apples vs E24)
    # --- three paired seeds for the v11 arm, so E24 vs E25 can be reported as mean +/- std ---
    # --- finish E24 first: seed 1, then seed 7 ---
    (24, 1,  "_1280_s1"),   #   E24 seed=1
    (24, 7,  "_1280_s7"),   #   E24 seed=7
    # --- (1) true ten-class VisDrone official-val anchor (corrected 2026-07-04): the old _vd10_960 actually
    #     ran as three-class, because ensure_data_yaml did not distinguish datasets and NMV_DATA was not passed
    #     to the subprocess. This uses the real ten-class root, run name _vis10_960, official val split,
    (1,  42, "_vis10_960", {"data": "visdrone10.yaml", "imgsz": 960, "eval_split": "val", "eval_imgsz": 960}),   # true VisDrone10 baseline
    (2,  42, "_vis10_960", {"data": "visdrone10.yaml", "imgsz": 960, "eval_split": "val", "eval_imgsz": 960}),   # true VisDrone10 +P2
    (17, 42, "_vis10_960", {"data": "visdrone10.yaml", "imgsz": 960, "eval_split": "val", "eval_imgsz": 960}),   # true VisDrone10 +P2+MCA+SA
    # --- (2) top the v8 headline (E01/E02) up to 3 seeds, so the +0.58/+0.67 pp can carry a paired t-test + 95% CI.
    #     Note: the overall gain sits inside the +/- std band and the t-test will most likely come out *not significant*;
    (1,  1,  "_1280_s1"),   #   E01 baseline seed=1  (checkpoint exists; resumed after the ten-class anchor)
    (1,  7,  "_1280_s7"),   #   E01 baseline seed=7
    (2,  1,  "_1280_s1"),   #   E02 +P2 seed=1
    (2,  7,  "_1280_s7"),   #   E02 +P2 seed=7
    # dim/depth ablations E20-23 are not in this 1280 queue; run them separately at 960, single seed, if needed.
]


def apply_run_suffix(exp):
    """If NMV_RUN_SUFFIX is set in env, return a copy of exp with the suffix
    appended to the run name. Used by run_multi_seed.py to keep seed runs
    isolated (e.g. E08_p2_mca_s1 vs E08_p2_mca)."""
    suffix = os.environ.get("NMV_RUN_SUFFIX", "")
    if not suffix:
        return exp
    out = dict(exp)
    out["name"] = exp["name"] + suffix
    return out


def banner(text, ch="="):
    print()
    print(ch * 78)
    print(f"  {text}")
    print(ch * 78)


def fmt_metrics(m):
    if not m:
        return "  (no metrics — training did not complete)"
    map_key = next((k for k in m if "mAP50" in k and "95" not in k), None)
    map95_key = next((k for k in m if "mAP50-95" in k or "mAP50_95" in k), None)
    p_key = next((k for k in m if k.startswith("metrics/precision")), None)
    r_key = next((k for k in m if k.startswith("metrics/recall")), None)
    parts = [f"epoch={m.get('epoch', '?')}"]
    if p_key:
        parts.append(f"P={m[p_key]:.4f}")
    if r_key:
        parts.append(f"R={m[r_key]:.4f}")
    if map_key:
        parts.append(f"mAP50={m[map_key]:.4f}")
    if map95_key:
        parts.append(f"mAP50-95={m[map95_key]:.4f}")
    return "  " + "  ".join(parts)


def run_experiment_in_process(exp):
    """Called inside the subprocess. Builds + trains/vals one experiment."""
    from ultralytics import YOLO

    exp = apply_run_suffix(exp)
    run_dir = RUNS / exp["name"]
    last_pt = run_dir / "weights" / "last.pt"
    hp = dict(HP)
    if os.environ.get("NMV_EPOCHS"):
        hp["epochs"] = int(os.environ["NMV_EPOCHS"])
        print(f"  [NMV_EPOCHS override] epochs={hp['epochs']}")
    if os.environ.get("NMV_WORKERS"):
        hp["workers"] = int(os.environ["NMV_WORKERS"])
        print(f"  [NMV_WORKERS override] workers={hp['workers']}")
    if os.environ.get("NMV_IMGSZ"):
        hp["imgsz"] = int(os.environ["NMV_IMGSZ"])
        print(f"  [NMV_IMGSZ override] imgsz={hp['imgsz']}")
    if os.environ.get("NMV_CACHE"):
        hp["cache"] = os.environ["NMV_CACHE"]
        print(f"  [NMV_CACHE override] cache={hp['cache']}")
    if os.environ.get("NMV_PATIENCE"):
        hp["patience"] = int(os.environ["NMV_PATIENCE"])
        print(f"  [NMV_PATIENCE override] patience={hp['patience']}")
    if os.environ.get("NMV_BATCH"):
        hp["batch"] = int(os.environ["NMV_BATCH"])
        print(f"  [NMV_BATCH override] batch={hp['batch']}")
    if os.environ.get("NMV_SEED"):
        hp["seed"] = int(os.environ["NMV_SEED"])
        print(f"  [NMV_SEED override] seed={hp['seed']}")

    if exp["kind"] == "train":
        if last_pt.exists() and not is_completed(run_dir):
            print(f"  Checkpoint found, resuming from {last_pt}")
            print(f"  resume overrides: workers={hp['workers']}, cache={hp['cache']!r}")
            model = YOLO(str(last_pt))
            model.train(resume=True, workers=hp["workers"], cache=hp["cache"])
        elif exp["cfg"]:
            model = YOLO(exp["cfg"]).load(str(WEIGHTS / exp["weights"]))
            model.train(
                data=str(DATA),
                project=str(RUNS),
                name=exp["name"],
                exist_ok=True,
                **hp,
            )
        else:
            model = YOLO(str(WEIGHTS / exp["weights"]))
            model.train(
                data=str(DATA),
                project=str(RUNS),
                name=exp["name"],
                exist_ok=True,
                **hp,
            )
    elif exp["kind"] == "val_only":
        src_run = RUNS / exp["source"] / "weights" / "best.pt"
        if not src_run.exists():
            print(f"[ERROR] Source weights not found: {src_run}")
            print("        Run the producing experiment first (e.g., --only 4 for E04).")
            sys.exit(2)
        model = YOLO(str(src_run))
        val_kwargs = dict(
            data=str(DATA),
            imgsz=hp["imgsz"],
            batch=hp["batch"],
            workers=0,
            project=str(RUNS),
            name=exp["name"],
            exist_ok=True,
            verbose=True,
            split="val",
        )
        val_kwargs.update(exp.get("val_kwargs", {}))
        model.val(**val_kwargs)
    else:
        raise ValueError(f"Unknown experiment kind: {exp['kind']}")


def run_with_watchdog(cmd, env, exp, run_dir):
    """subprocess.Popen plus a heartbeat poll watchdog (added 2026-06-10, after two overnight hangs).

The hang looks like this: the child process never exits, GPU utilisation sits at ~2%, and results.csv
stops growing for hours. A blocking subprocess.run never sees an exit code, so the existing crash-retry
never fires. Here the mtime of results.csv / last.pt is the heartbeat; once it stalls past the threshold
the process is killed and a synthetic returncode is returned, so the caller takes the existing resume path

    - train kind: the first heartbeat gets a grace period of NMV_WATCHDOG_FIRST_MIN (default 120 min, to allow
      building the disk cache and a slow first epoch); after that, a stall longer than NMV_WATCHDOG_MIN
      (default 90 min, three times the slowest 26 min/epoch) counts as a hang.
    - val_only kind: no results.csv heartbeat, so NMV_WATCHDOG_MIN is a hard total timeout.
    """
    POLL_SEC = 60
    # An epoch at 1280/batch1 takes ~20-23 min, so the stall threshold allows about 2 epochs. Lowered from
    # 90/120 to 45/60 (defaults) on 2026-06-23, along with a fix for resume wrongly taking the first-heartbeat grace (see the branch below).
    stale_limit = float(os.environ.get("NMV_WATCHDOG_MIN", "45")) * 60
    first_limit = float(os.environ.get("NMV_WATCHDOG_FIRST_MIN", "60")) * 60
    heartbeat_files = [run_dir / "results.csv", run_dir / "weights" / "last.pt"]

    def latest_heartbeat():
        ts = [f.stat().st_mtime for f in heartbeat_files if f.exists()]
        return max(ts) if ts else None

    proc = subprocess.Popen(cmd, env=env)
    start = time.time()
    while True:
        rc = proc.poll()
        if rc is not None:
            return rc
        time.sleep(POLL_SEC)
        now = time.time()
        if exp["kind"] == "train":
            hb = latest_heartbeat()
            if hb is None:
                # cold start, no checkpoint on disk yet: the first epoch plus building the disk cache can be slow
                stalled, limit = now - start, first_limit
            elif hb < start:
                # resume: the checkpoint is from last time and there is no new heartbeat yet this run. The model is
                # warm, so the first resumed epoch is about as fast as a normal one - use the normal stall threshold
                # rather than the first-heartbeat grace. (Old bug: every resume reset the grace to 120 min, so a
                stalled, limit = now - start, stale_limit
            else:
                stalled, limit = now - hb, stale_limit
        else:
            stalled, limit = now - start, stale_limit
        if stalled > limit:
            print(f"[WATCHDOG] {exp['name']} no heartbeat for {stalled/60:.0f} min "
                  f"(limit {limit/60:.0f} min) — killing pid {proc.pid}, "
                  f"will go through normal crash-retry/resume path")
            proc.kill()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)])
            return -888  # synthetic returncode; the caller treats it as a crash


def _epochs_logged(run_dir):
    """Number of epochs already written to results.csv (progress signal; monotonic across resumes). Returns 0 if unreadable."""
    f = run_dir / "results.csv"
    if not f.exists():
        return 0
    try:
        with open(f, encoding="utf-8") as fh:
            return max(0, sum(1 for line in fh if line.strip()) - 1)  # drop the header row
    except Exception:
        return 0


def dispatch_one(exp, force=False):
    """Run one experiment in a subprocess with crash-resume retry.

    Returns metrics dict on success. Progress-aware retry: as long as each attempt writes at least one
    new epoch to disk the failure budget resets; only MAX_ATTEMPTS consecutive attempts with zero epoch
    progress call sys.exit(1). (So a frequent but recoverable hang - one that stalls every few epochs -
    Resume relies on early_stop_resume.py preserving optimizer and epoch state in last.pt.
    """
    exp = apply_run_suffix(exp)
    run_dir = RUNS / exp["name"]
    last_pt = run_dir / "weights" / "last.pt"

    if exp["kind"] == "train" and not force:
        if is_completed(run_dir):
            print(f"[SKIP] {exp['name']} already completed.")
            return read_best(run_dir / "results.csv")
        if last_pt.exists():
            print(f"[RESUME] {exp['name']} — checkpoint found, will resume")

    env = os.environ.copy()
    # Windows does not support expandable_segments
    # max_split_size_mb:64 -> smaller than 128, which reduces reserved blocks that are held but unused
    # garbage_collection_threshold:0.6 -> Windows shared GPU memory inflates the apparent ceiling to ~16 GB,
    #   so 0.8 fires too late (already OOM); 0.6 brings it forward to ~9.6 GB
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:64,garbage_collection_threshold:0.6",
    )
    for k, v in exp.get("env", {}).items():
        env[k] = v
        print(f"  env: {k}={v}")

    cmd = [sys.executable, str(__file__), "--exp-idx", str(exp["idx"])]

    attempt = 0
    fails_without_progress = 0
    epochs_before = _epochs_logged(run_dir)   # epochs already on disk on entry (>0 when resuming)
    while True:
        attempt += 1
        if attempt > 1:
            banner(f"[{exp['idx']}] {exp['name']} — RETRY (attempt {attempt}, "
                   f"{fails_without_progress}/{MAX_ATTEMPTS} consecutive attempts without progress, resume from last.pt)", "!")
            if exp["kind"] == "train" and not last_pt.exists():
                print(f"[FATAL] No last.pt at {last_pt}, cannot resume. Aborting.")
                sys.exit(1)
            # the retry goes straight to workers=0: no child processes, which avoids the two failure modes where
            # importing torch during spawn requests virtual memory and raises WinError 1455 or a numpy ArrayMemoryError
            env["NMV_WORKERS"] = "0"
            print(f"  retry override: NMV_WORKERS=0 (no DataLoader spawn — saves ~2GB virtual memory)")
            # give Windows time to reclaim the working set, standby memory and page-file commit of the crashed child
            print(f"  sleeping 30s for OS to reclaim memory from crashed subprocess...")
            time.sleep(30)

        print(f"  cmd: {' '.join(cmd)}  (attempt {attempt})")
        t0 = datetime.now()
        rc = run_with_watchdog(cmd, env, exp, run_dir)
        elapsed = datetime.now() - t0
        print(f"  -> exit code {rc}, elapsed {elapsed}")

        if rc == 0:
            break

        # rc!=0 but run_dir is already completed (ultralytics occasionally exits non-zero after a natural
        # early stop); treat that as success
        if exp["kind"] == "train" and is_completed(run_dir):
            print(f"  [INFO] rc={rc} but run_dir is_completed — treating as success")
            break

        # progress-aware retry budget: if this attempt wrote a new epoch to disk, treat it as progress and
        # reset the failure count, so a hang that recurs every few epochs still resumes to completion; only
        # MAX_ATTEMPTS consecutive attempts with zero epoch progress count as a real hang and call sys.exit
        epochs_now = _epochs_logged(run_dir)
        if epochs_now > epochs_before:
            print(f"[PROGRESS] {exp['name']} rc={rc} but advanced {epochs_before}->{epochs_now} "
                  f"epoch(s) - resetting the no-progress failure count (was {fails_without_progress})")
            fails_without_progress = 0
            epochs_before = epochs_now
        else:
            fails_without_progress += 1
            print(f"[CRASH] {exp['name']} attempt {attempt} rc={rc}, no progress (epochs stuck at "
                  f"{epochs_now}); {fails_without_progress}/{MAX_ATTEMPTS} consecutive without progress")
            if fails_without_progress >= MAX_ATTEMPTS:
                print(f"[FATAL] {exp['name']} made no epoch progress {MAX_ATTEMPTS} times in a row; treating it as a real hang. Stopping pipeline.")
                print(f"        Inspect {run_dir} and re-run with: "
                      f"python scripts/train.py --only {exp['idx']}")
                sys.exit(1)

    if exp["kind"] == "train":
        return read_best(run_dir / "results.csv")
    return None


def auto_eval_test(name, data_yaml=None, split="test", imgsz=1280, batch=2):
    """After each main-line model finishes training (or is skipped as already complete), evaluate on test at imgsz=1280:
      (1) standard mAP (overall mAP@0.5 / mAP@0.5:0.95 / per-class) - via val.py
      (2) COCO size buckets (mAP_small/medium/large) - via eval_size_buckets.py
    Each is idempotent (skipped when results exist), so re-runs are fast; the buckets are non-fatal (a failure does not stop the queue).
    Note: when (1) already exists we must not return early, or (2) would never be backfilled for completed models."""
    bp = RUNS / name / "weights" / "best.pt"
    if not bp.exists():
        print(f"  [skip eval] best.pt does not exist (training unfinished?): {bp}")
        return

    # (1) standard test evaluation
    data_yaml = data_yaml or (ROOT / "configs" / "data" / "nmv_visdrone_3cls.yaml")
    out = RUNS / f"{name}_eval_{split}"
    if (out / "results.csv").exists():
        print(f"  [skip eval] {out.name} already exists")
    else:
        print(f"  --- {split} evaluation of {name} @imgsz={imgsz} ---")
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "val.py"),
             "--run", name, "--data", str(data_yaml),
             "--split", split, "--imgsz", str(imgsz), "--batch", str(batch)]
        ).returncode
        print(f"  -> eval rc={rc}")

    # (2) COCO size buckets (the mAP_small figures the paper turns on) - idempotent and non-fatal
    bdir = RUNS / f"{name}_eval_{split}_buckets"
    if (bdir / "metrics.json").exists():
        print(f"  [skip buckets] {bdir.name} already exists")
        return
    print(f"  --- size-bucket evaluation of {name} @imgsz={imgsz} split={split} ---")
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "eval_size_buckets.py"),
             "--run", name, "--data", str(data_yaml),
             "--split", split, "--imgsz", str(imgsz)],
            check=False,
        )
    except Exception as e:
        print(f"  [warn] bucket eval failed (non-fatal): {e}")


def _unpack_queue_entry(entry):
    """A PAPER_QUEUE entry is either a 3-tuple (idx, seed, suffix) or a 4-tuple
    (idx, seed, suffix, override), where override may contain:
      {'data': 'xxx.yaml', 'imgsz': 960, 'eval_split': 'val', 'eval_imgsz': 960}
    """
    idx, seed, suffix = entry[:3]
    ov = entry[3] if len(entry) > 3 else {}
    return idx, seed, suffix, ov


def run_paper_queue(force=False):
    """Bare-run entry point: train the paper 1280 main line and evaluate on test automatically."""
    global DATA
    banner("paper main line (imgsz=1280, batch=2) - training + test evaluation", "#")
    base_data = DATA
    base_data_env = os.environ.get("NMV_DATA")
    base_imgsz_env = os.environ.get("NMV_IMGSZ")
    print("  Queue (each ~56h measured at 1280/b2; old results are untouched, everything goes to new suffixed directories):")
    for entry in PAPER_QUEUE:
        idx, seed, suffix, ov = _unpack_queue_entry(entry)
        tag = f"  [{ov['data']}@{ov.get('imgsz', '-')}]" if ov.get("data") else ""
        print(f"    {EXPERIMENTS_BY_IDX[idx]['name']}{suffix:12s}  seed={seed}{tag}")
    print("  Ctrl+C any time; running `python scripts/train.py` again resumes from the checkpoint.\n")

    results = []
    t0 = datetime.now()
    for entry in PAPER_QUEUE:
        idx, seed, suffix, ov = _unpack_queue_entry(entry)
        # --- per-entry dataset / imgsz overrides; restore the defaults when absent, to keep entries isolated ---
        if ov.get("data"):
            DATA = ensure_data_yaml(Path(ROOT / "configs" / "data" / ov["data"]))
            os.environ["NMV_DATA"] = str(DATA)
        else:
            DATA = base_data
            if base_data_env is not None:
                os.environ["NMV_DATA"] = base_data_env
            else:
                os.environ.pop("NMV_DATA", None)
        if ov.get("imgsz"):
            os.environ["NMV_IMGSZ"] = str(ov["imgsz"])
        elif base_imgsz_env is not None:
            os.environ["NMV_IMGSZ"] = base_imgsz_env
        else:
            os.environ.pop("NMV_IMGSZ", None)
        os.environ["NMV_RUN_SUFFIX"] = suffix
        if seed != 42:
            os.environ["NMV_SEED"] = str(seed)
        else:
            os.environ.pop("NMV_SEED", None)
        exp = EXPERIMENTS_BY_IDX[idx]
        name = exp["name"] + suffix
        banner(f"{name}  (seed={seed}) — {exp['desc']}", "#")
        m = dispatch_one(exp, force=force)
        results.append((name, m))
        print(fmt_metrics(m))
        eval_split = ov.get("eval_split", "test")
        eval_imgsz = int(ov.get("eval_imgsz", ov.get("imgsz", 1280)))
        eval_batch = int(ov.get("eval_batch", 2))
        auto_eval_test(name, data_yaml=DATA, split=eval_split, imgsz=eval_imgsz, batch=eval_batch)

    DATA = base_data  # restore the global, so later calls are not polluted
    if base_data_env is not None:
        os.environ["NMV_DATA"] = base_data_env
    else:
        os.environ.pop("NMV_DATA", None)

    banner(f"main line finished - elapsed {datetime.now() - t0}", "#")
    print(f"  {'name':<28}{'mAP50':>10}{'mAP50-95':>12}")
    print("  " + "-" * 50)
    for name, m in results:
        if not m:
            print(f"  {name:<28}  (no metrics)"); continue
        mk = next((k for k in m if "mAP50" in k and "95" not in k), None)
        m9 = next((k for k in m if "mAP50-95" in k or "mAP50_95" in k), None)
        f = lambda k: f"{m[k]:.4f}" if k and k in m else "-"
        print(f"  {name:<28}{f(mk):>10}{f(m9):>12}")
    print("\n  Next steps (optional, no training needed; one command each):")
    print("    python scripts/sahi_predict.py --run E08_p2_mca_1280 --split test")
    print("    python scripts/sahi_predict.py --run E01_baseline_1280 --split test")
    print("    python scripts/aggregate_seeds.py     # 3-seed mean +/- std and paired tests")
    print("    python scripts/export_table.py        # main comparison table (includes 960-vs-1280)")
    print(f"\n  Output root: {RUNS}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=int, default=None,
                   help="Only run the experiment with this idx (1-26)")
    p.add_argument("--skip", type=int, nargs="+", default=[],
                   help="Skip these experiment idxs (added to DEFAULT_SKIP)")
    p.add_argument("--start", type=int, default=0,
                   help="Start from this idx (skip earlier)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if already completed")
    p.add_argument("--include-default-skip", action="store_true",
                   help="include the experiments skipped by DEFAULT_SKIP (DEFAULT_SKIP applies unless --only is used)")
    p.add_argument("--exp-idx", type=int, default=None,
                   help=argparse.SUPPRESS)  # internal: subprocess entry point
    args = p.parse_args()

    if args.exp_idx is not None:
        exp = next(e for e in EXPERIMENTS if e["idx"] == args.exp_idx)
        run_experiment_in_process(exp)
        return

    # A bare run (no --only / --start / --skip / --include-default-skip) means the paper 1280 main line.
    #   To run the old EXPERIMENTS list, pass --only/--start/--skip explicitly.
    if (args.only is None and args.start == 0 and not args.skip
            and not args.include_default_skip):
        run_paper_queue(force=args.force)
        return

    # --only bypasses DEFAULT_SKIP (the user named an idx, so run it)
    apply_default_skip = (
        args.only is None and not args.include_default_skip
    )
    if apply_default_skip:
        skipped_by_default = [i for i in DEFAULT_SKIP if i not in args.skip]
        if skipped_by_default:
            print("Default-skipped (use --only <idx> or --include-default-skip to force):")
            for i in skipped_by_default:
                print(f"  [{i}] {DEFAULT_SKIP_REASONS[i]}")

    todo = [
        e for e in EXPERIMENTS
        if (args.only is None or e["idx"] == args.only)
        and e["idx"] >= args.start
        and e["idx"] not in args.skip
        and (not apply_default_skip or e["idx"] not in DEFAULT_SKIP)
    ]
    if not todo:
        print("No experiments to run.")
        return

    banner(f"Non-motor vehicle SOD experiments — {len(todo)} to run", "#")
    for e in todo:
        print(f"  [{e['idx']}] {e['name']:25s} {e['desc']}")

    results = []
    t0 = datetime.now()
    for exp in todo:
        banner(f"[{exp['idx']}] {exp['name']} — {exp['desc']}", "#")
        m = dispatch_one(exp, force=args.force)
        results.append((exp, m))
        print(fmt_metrics(m))

    banner(f"Final summary — total elapsed {datetime.now() - t0}", "#")
    print(f"  {'idx':<5}{'name':<25}{'mAP50':>10}{'mAP50-95':>12}{'P':>10}{'R':>10}")
    print("  " + "-" * 72)
    for exp, m in results:
        if m is None:
            print(f"  {exp['idx']:<5}{exp['name']:<25}  (no metrics)")
            continue
        map_key = next((k for k in m if "mAP50" in k and "95" not in k), None)
        map95_key = next((k for k in m if "mAP50-95" in k or "mAP50_95" in k), None)
        p_key = next((k for k in m if k.startswith("metrics/precision")), None)
        r_key = next((k for k in m if k.startswith("metrics/recall")), None)
        def fmt(k):
            return f"{m[k]:.4f}" if k and k in m else "-"
        print(f"  {exp['idx']:<5}{exp['name']:<25}{fmt(map_key):>10}{fmt(map95_key):>12}{fmt(p_key):>10}{fmt(r_key):>10}")
    print()
    print(f"  Outputs: {RUNS}")


if __name__ == "__main__":
    main()
