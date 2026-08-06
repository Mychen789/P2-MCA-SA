"""One-shot runner for the §5.7 held-out robustness evaluation.

Runs eval_cross_dataset.py for both the v8 hero (E16') and the v11 hero (E24)
on the same-source held-out set (visdrone_full_nmv3), so the user only pastes
one short command instead of two long, paste-fragile ones.

Usage from the repository root:
  python scripts/run_heldout_eval.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts" / "eval_cross_dataset.py"
DATA = "configs/data/visdrone_full_nmv3.yaml"

RUNS = [
    ("E16_mca_scale_assign_hi32_1280", "v8 hero E16'"),
    ("E24_yolo11m_p2_mca_scale_1280", "v11 hero E24"),
]

for run, label in RUNS:
    print(f"\n########## {label}  ({run}) ##########", flush=True)
    cmd = [
        sys.executable, str(EVAL),
        "--run", run,
        "--data", DATA,
        "--name", "visdrone_full",
        "--split", "val",
        "--imgsz", "960",
        "--batch", "4",
        "--iou", "0.7",
    ]
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc != 0:
        print(f"[WARN] {run} exited with code {rc}", flush=True)

print("\n=== both held-out evals done; see runs/_summary/cross_dataset_eval.csv ===", flush=True)
