"""Batch wrapper to train E01 / E08 / E11 with 4 additional seeds and then
auto-evaluate each new run on the test split.

Why this exists:
  - 北大核心审稿人对单 seed 结果普遍不接受。
  - 原 seed=42 已经训过 (runs/E01_baseline, E08_p2_mca, E11_full_mca_cagfpn)；
    本脚本补充 seed=1,7,123,2026，4 个新种子 × 3 实验 = 12 次训练，
    在 RTX 5060 / imgsz=960 / batch=4 下每次 ~15h，合计 ~7.5 天连续。
  - 每次训练完成立即跑 test 评估，避免训完才发现需要补 eval。

Usage:
  cd P2-MCA-SA
  python scripts/run_multi_seed.py                 # 跑全部 12 个 (训 + 评)
  python scripts/run_multi_seed.py --only-seed 1   # 只跑 seed=1 的 3 个
  python scripts/run_multi_seed.py --only-exp 8    # 只跑 E08 的 4 个 seed
  python scripts/run_multi_seed.py --skip-eval     # 只训不评 (eval 留到后面批量做)

After completion run:
  python scripts/aggregate_seeds.py    # 聚合 mean ± std
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

# 论文优化版 (imgsz=1280)。headline 双模型只补 2 个 seed (42 已由 run_paper 跑),
# 共报告 3 seed 的 mean ± std；E11/CAGFPN 已退出主线(掉 mAP50-95)故不再跑。
# 推荐用 scripts/run_paper.py 统一驱动全队列;本脚本作为单独补 seed 的备用入口。
SEEDS = [1, 7]
EXPS = [
    (1, "E01_baseline"),
    (8, "E08_p2_mca"),
]


def run_train(idx, name, seed):
    suffix = f"_1280_s{seed}"  # 与 run_paper.py / aggregate_seeds.py 的 1280 命名一致
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
    suffix = f"_1280_s{seed}"  # 与 run_paper.py / aggregate_seeds.py 的 1280 命名一致
    run_dir_name = name + suffix
    bp = ROOT / "runs" / run_dir_name / "weights" / "best.pt"
    if not bp.exists():
        print(f"  [SKIP EVAL] best.pt not found: {bp}")
        return False
    print(f"\n--- EVAL  {run_dir_name}  on test split ---")
    cmd = [PY, str(ROOT / "scripts" / "val.py"), "--run", run_dir_name,
           "--split", "test", "--imgsz", "1280"]  # 必须 1280 评估,与训练分辨率一致
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
    print(f"\nEstimated wallclock: ~{total * 56}h ≈ {total * 56 / 24:.1f} days (imgsz=1280, batch=2, 实测 ~56h/run)")
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
