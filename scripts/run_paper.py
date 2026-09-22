"""Single-entry orchestrator for the paper configuration (imgsz=1280) - run just this one script.

Usage (PowerShell):
  conda activate yolov8
  cd P2-MCA-SA
  python scripts/sanity_check.py        # first confirm the patches and model parsing are fine
  python scripts/run_paper.py           # run the full priority queue (Ctrl+C any time; re-running resumes)

Executed in priority order (most important first; --until stops after a given phase):
  headline    : retrain E08 (P2+MCA) and E01 (baseline), one seed=42 each -> test eval -> SAHI / size buckets
  ablation    : add E02 (+P2) seed=42, completing the minimal fair ablation baseline / +P2 / +P2+MCA
  seeds       : add seeds 1 and 7 to E08 and E01 (3 seeds each), giving mean +/- std and paired tests
  components  : MCA component ablation on the same mca.yaml, EMA-only / CoordAtt-only (NMV_MCA_MODE)

Design notes:
  - Every new 1280 run carries a `_1280[_sN]` run suffix and does not overwrite the old 960 results (kept as a resolution ablation).
  - Training is delegated to train.py (which retries on crash and resumes); this script skips eval/SAHI/buckets idempotently.
  - Evaluation is always at imgsz=1280 (matching the training resolution, or the whole table is void).
  - SAHI always uses conf=0.01, slice=640 (too high a conf truncates the PR curve and depresses mAP_small).

Interruptible: after Ctrl+C, re-running `python scripts/run_paper.py` continues from the checkpoint (guaranteed by
the early_stop_resume patch in train.py). Each train run at 1280/batch2 takes about 56 h (smoke test measured 22.4 min/epoch).

Other common arguments:
  --until headline|ablation|seeds|components|all   phase to stop after (default all)
  --batch N         override the training batch (default is train.py HP=2; try 3 if VRAM allows)
  --smoke           run only a 2-epoch smoke test (check the current batch does not OOM at 1280) and exit
  --dry-run         print the queue and the cumulative time estimate without executing
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN = str(ROOT / "scripts" / "train.py")
VAL = str(ROOT / "scripts" / "val.py")
SAHI = str(ROOT / "scripts" / "sahi_predict.py")
BUCKETS = str(ROOT / "scripts" / "eval_size_buckets.py")
RUNS = ROOT / "runs"

PHASE_RANK = {"headline": 1, "ablation": 2, "seeds": 3, "components": 4, "all": 99}

# idx in the train.py EXPERIMENTS table -> base run name (apply_run_suffix appends NMV_RUN_SUFFIX)
EXP_BASENAME = {1: "E01_baseline", 2: "E02_p2", 8: "E08_p2_mca"}


def banner(text, ch="#"):
    print("\n" + ch * 80 + f"\n  {text}\n" + ch * 80, flush=True)


def run_name(idx, suffix):
    return EXP_BASENAME[idx] + suffix


def _run(cmd, extra_env=None, label=""):
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    t0 = datetime.now()
    print(f"  $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    dt = datetime.now() - t0
    print(f"  -> rc={rc}  elapsed {dt}  [{label}]", flush=True)
    return rc


def do_train(idx, suffix, seed=42, mca_mode=None, batch=None, dry=False):
    """Retrain one experiment. train.py brings its own is_completed skip, crash retry and resume."""
    name = run_name(idx, suffix)
    extra = {"NMV_RUN_SUFFIX": suffix}
    if seed != 42:
        extra["NMV_SEED"] = seed
    if mca_mode:
        extra["NMV_MCA_MODE"] = mca_mode
    if batch:
        extra["NMV_BATCH"] = batch
    banner(f"TRAIN  {name}   (idx={idx} seed={seed}"
           + (f" mca_mode={mca_mode}" if mca_mode else "") + ")")
    if dry:
        print(f"  [dry-run] would train {name}")
        return name
    _run([PY, TRAIN, "--only", str(idx)], extra_env=extra, label=f"train {name}")
    return name


def do_eval(name, dry=False):
    """Test-set evaluation at imgsz=1280. Idempotent: skipped when results.csv already exists."""
    out = RUNS / f"{name}_eval_test"
    if (out / "results.csv").exists():
        print(f"  [skip eval] {out.name} already exists")
        return
    if dry:
        print(f"  [dry-run] would eval {name} @1280")
        return
    _run([PY, VAL, "--run", name, "--split", "test", "--imgsz", "1280", "--batch", "2"],
         label=f"eval {name}")


def do_sahi(name, dry=False):
    """SAHI sliced inference (slice=640, conf=0.01). Idempotent: skipped when metrics.json already exists."""
    out = RUNS / f"E07_sahi_{name}_test"
    if (out / "metrics.json").exists():
        print(f"  [skip SAHI] {out.name} already exists")
        return
    if dry:
        print(f"  [dry-run] would SAHI {name}")
        return
    _run([PY, SAHI, "--run", name, "--split", "test",
          "--slice", "640", "--overlap", "0.2", "--conf", "0.01"], label=f"sahi {name}")


def do_buckets(name, dry=False):
    """COCO size-bucket mAP_small/medium/large (unsliced, conf=0.001). Idempotently skipped."""
    out = RUNS / f"{name}_eval_test_buckets"
    if (out / "metrics.json").exists():
        print(f"  [skip buckets] {out.name} already exists")
        return
    if dry:
        print(f"  [dry-run] would size-bucket eval {name}")
        return
    _run([PY, BUCKETS, "--run", name, "--split", "test", "--imgsz", "1280"],
         label=f"buckets {name}")


def smoke_test(batch, dry=False):
    """2-epoch smoke test: run E08 (the heaviest, +P2+MCA) at 1280 with the given batch to check it does not OOM."""
    banner(f"SMOKE TEST  E08 @1280 batch={batch or 'default (2)'}  2 epochs", "!")
    if dry:
        print("  [dry-run] would run 2-epoch smoke test"); return
    extra = {"NMV_RUN_SUFFIX": "_1280_smoke", "NMV_EPOCHS": "2"}
    if batch:
        extra["NMV_BATCH"] = batch
    rc = _run([PY, TRAIN, "--only", "8", "--force"], extra_env=extra, label="smoke")
    if rc == 0:
        print("\n[OK] smoke test passed - this batch does not OOM at 1280, the real queue can start.")
        print("     For the real run drop --smoke: python scripts/run_paper.py")
    else:
        print("\n[FAIL] smoke test failed (most likely OOM). Lower the batch (--batch 2) and retry.")
    return rc


def main():
    p = argparse.ArgumentParser(description="single-entry orchestrator for the paper configuration (imgsz=1280)")
    p.add_argument("--until", choices=list(PHASE_RANK), default="all",
                   help="phase to stop after (default all)")
    p.add_argument("--batch", type=int, default=None,
                   help="override the training batch (default is train.py HP=2)")
    p.add_argument("--smoke", action="store_true", help="run only the 2-epoch smoke test, then exit")
    p.add_argument("--dry-run", action="store_true", help="print the queue without executing")
    args = p.parse_args()

    batch = args.batch
    dry = args.dry_run

    if args.smoke:
        smoke_test(batch, dry=dry)
        return

    rank = PHASE_RANK[args.until]
    t_global = datetime.now()
    banner(f"PAPER QUEUE @imgsz=1280  until={args.until}  batch={batch or 'HP default (2)'}"
           + ("  [DRY-RUN]" if dry else ""))
    print("  Each train run at 1280/batch2 takes about 28h; eval ~10min; SAHI ~30min.")
    print("  Ctrl+C any time; re-running this script resumes.\n")
    print("  Cumulative time by phase: headline ~2.4d | +ablation ~3.6d | +3seed ~7d | +components ~10d")

    # ---------------- Phase 1: headline (E08 + E01, seed 42) ----------------
    if rank >= PHASE_RANK["headline"]:
        banner("PHASE 1 / headline  (E08 P2+MCA  vs  E01 baseline, seed=42)", "=")
        e08 = do_train(8, "_1280", seed=42, batch=batch, dry=dry)
        e01 = do_train(1, "_1280", seed=42, batch=batch, dry=dry)
        for n in (e08, e01):
            do_eval(n, dry=dry); do_buckets(n, dry=dry); do_sahi(n, dry=dry)
        print("\n[MILESTONE 1] headline done: at 1280, the standard mAP / "
              "mAP_small (SAHI + buckets) / per-class AP for baseline vs P2+MCA are ready - single seed, enough for Table 1.")

    # ---------------- Phase 2: ablation middle row (E02 +P2) ----------------
    if rank >= PHASE_RANK["ablation"]:
        banner("PHASE 2 / ablation  (E02 +P2, seed=42 - separates the P2 head from MCA)", "=")
        e02 = do_train(2, "_1280", seed=42, batch=batch, dry=dry)
        do_eval(e02, dry=dry); do_buckets(e02, dry=dry); do_sahi(e02, dry=dry)
        print("\n[MILESTONE 2] minimal fair ablation done: baseline / +P2 / +P2+MCA, three rows on one recipe.")

    # ---------------- Phase 3: multi-seed (E08, E01 × seed 1,7) -------------
    if rank >= PHASE_RANK["seeds"]:
        banner("PHASE 3 / seeds  (add seeds 1 and 7 to E08 and E01, 3 seeds each)", "=")
        for seed in (1, 7):
            for idx in (8, 1):
                n = do_train(idx, f"_1280_s{seed}", seed=seed, batch=batch, dry=dry)
                do_eval(n, dry=dry)
        print("\n[MILESTONE 3 / suggested stopping point] 3 seeds complete. Run aggregate_seeds.py for mean +/- std and the paired tests.")

    # ---------------- Phase 4: MCA component ablation -----------------------
    if rank >= PHASE_RANK["components"]:
        banner("PHASE 4 / components  (same mca.yaml: EMA-only / CoordAtt-only)", "=")
        n_ema = do_train(8, "_1280_emaonly", seed=42, mca_mode="ema", batch=batch, dry=dry)
        do_eval(n_ema, dry=dry); do_buckets(n_ema, dry=dry)
        n_ca = do_train(8, "_1280_caonly", seed=42, mca_mode="ca", batch=batch, dry=dry)
        do_eval(n_ca, dry=dry); do_buckets(n_ca, dry=dry)
        print("\n[done] component ablation: EMA-only / CoordAtt-only / MCA-full (=E08_p2_mca_1280), three-way comparison.")

    banner(f"DONE  until={args.until}  total elapsed {datetime.now() - t_global}")
    print("  Generate the paper material (once training and evaluation have finished):")
    print("    python scripts/export_table.py        # main comparison table (reads *_1280*_eval_test)")
    print("    python scripts/aggregate_seeds.py     # 3-seed mean +/- std and paired tests")
    print("    python scripts/per_class_ap.py        # per-class AP")
    print("    python scripts/plot_curves.py / plot_pareto.py / plot_detections.py")
    print(f"\n  Output root: {RUNS}")


if __name__ == "__main__":
    main()
