"""Batch wrapper to train E01 / E08 / E11 with 4 additional seeds and then
auto-evaluate each new run on the test split.

Why this exists:
  - Single-seed results are generally not accepted by reviewers.
  - seed=42 has already been trained (runs/E01_baseline, E08_p2_mca, E11_full_mca_cagfpn);
    this script adds seeds 1, 7, 123 and 2026 - 4 new seeds x 3 experiments = 12 runs,
    about 15 h each on an RTX 5060 at imgsz=960 / batch=4, roughly 7.5 days back to back.
  - Each run is evaluated on test immediately after training, so no run finishes only to need a later eval.

Usage:
  cd P2-MCA-SA
  python scripts/run_multi_seed.py                 # run all 12 (train + eval)
  python scripts/run_multi_seed.py --only-seed 1   # only the 3 runs with seed=1
  python scripts/run_multi_seed.py --only-exp 8    # only the 4 seeds of E08
  python scripts/run_multi_seed.py --skip-eval     # train only (evaluate later in a batch)

After completion run:
  python scripts/aggregate_seeds.py    # aggregate mean +/- std
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

# Paper configuration (imgsz=1280). The two headline models need only 2 extra seeds (42 already
# came from run_paper), for 3 seeds of mean +/- std in total; E11/CAGFPN left the main line (it lost mAP50-95).
# scripts/run_paper.py drives the whole queue; this script is the standalone entry for topping up seeds.
SEEDS = [1, 7]
EXPS = [
    (1, "E01_baseline"),
    (8, "E08_p2_mca"),
]


def run_train(idx, name, seed):
    suffix = f"_1280_s{seed}"  # matches the 1280 naming of run_paper.py / aggregate_seeds.py
    print(f"\n{'#' * 78}")
    print(f"#  TRAIN  {name}{suffix}   seed={seed}   started {datetime.now()}")
    print(f"{'#' * 78}")
    env = os.environ.copy()
    env["NMV_SEED"] = str(seed)
    env["NMV_RUN_SUFFIX"] = suffix
    cmd = [PY, str(ROOT / "scripts" / "train.py"), "--only", str(idx)]
    t0 = datetime.now()
    rc = subprocess.run(cmd, env=env).returncode
    elapsed = datetime.now() - t0
    print(f"  -> train rc={rc}, elapsed {elapsed}")
    return rc == 0


def run_eval(name, seed):
    suffix = f"_1280_s{seed}"  # matches the 1280 naming of run_paper.py / aggregate_seeds.py
    run_dir_name = name + suffix
    bp = ROOT / "runs" / run_dir_name / "weights" / "best.pt"
    if not bp.exists():
        print(f"  [SKIP EVAL] best.pt not found: {bp}")
        return False
    print(f"\n--- EVAL  {run_dir_name}  on test split ---")
    cmd = [PY, str(ROOT / "scripts" / "val.py"), "--run", run_dir_name,
           "--split", "test", "--imgsz", "1280"]  # must evaluate at 1280, matching the training resolution
    rc = subprocess.run(cmd).returncode
    return rc == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only-seed", type=int, default=None,
                   help="Only run this seed (e.g. 1 / 7 / 123 / 2026)")
    p.add_argument("--only-exp", type=int, default=None,
                   help="Only run this experiment idx (1 / 8 / 11)")
    p.add_argument("--skip-eval", action="store_true",
                   help="Train only, skip the per-run test evaluation")
    args = p.parse_args()

    seeds = [args.only_seed] if args.only_seed is not None else SEEDS
    exps = [e for e in EXPS if args.only_exp is None or e[0] == args.only_exp]

    if not seeds or not exps:
        print("Nothing to run."); return

    total = len(seeds) * len(exps)
    print(f"=== Multi-seed plan: {total} train runs ({len(exps)} exps × {len(seeds)} seeds) ===")
    for s in seeds:
        for idx, name in exps:
            print(f"  - {name}_s{s}")
    print(f"\nEstimated wallclock: ~{total * 56}h = {total * 56 / 24:.1f} days (imgsz=1280, batch=2, measured ~56h/run)")
    print(f"Started at {datetime.now()}\n")

    t_global = datetime.now()
    failures = []
    for seed in seeds:
        for idx, name in exps:
            ok = run_train(idx, name, seed)
            if not ok:
                failures.append((name, seed, "train"))
                continue
            if not args.skip_eval:
                ok = run_eval(name, seed)
                if not ok:
                    failures.append((name, seed, "eval"))
            time.sleep(5)  # let GPU memory drain

    elapsed = datetime.now() - t_global
    print(f"\n{'#' * 78}")
    print(f"#  DONE  total elapsed {elapsed}")
    print(f"#  failures: {len(failures)}")
    for name, seed, stage in failures:
        print(f"    - {name}_s{seed}  ({stage})")
    print(f"{'#' * 78}")
    print("\nNext: python scripts/aggregate_seeds.py")


if __name__ == "__main__":
    main()
