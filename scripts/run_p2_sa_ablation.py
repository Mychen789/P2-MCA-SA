"""Run the missing P2+SA isolated ablation from a VS Code terminal.

This wrapper deliberately does not modify PAPER_QUEUE. It launches experiment
E26 from scripts/train.py with an isolated run name, enables TAL assignment
auditing, and evaluates the resulting best.pt with both Ultralytics metrics and
COCO small/medium/large buckets.

Recommended order:
  python scripts/run_p2_sa_ablation.py --stage screen
  python scripts/run_p2_sa_ablation.py --stage confirm
  python scripts/run_p2_sa_ablation.py --stage vis10

Training is resumable. Re-run the same command after Ctrl+C to continue from
last.pt. Use --dry-run to inspect commands without starting the GPU job.
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN = ROOT / "scripts" / "train.py"
VAL = ROOT / "scripts" / "val.py"
BUCKETS = ROOT / "scripts" / "eval_size_buckets.py"
RUNS = ROOT / "runs"
BASE_NAME = "E26_p2_scale_assign_hi32"


STAGES = {
    # Cheap directional screen before committing to three long 1280 runs.
    "screen": [
        dict(idx=26, base=BASE_NAME, scale_assign=True,
             suffix="_screen960", seed=42, imgsz=960,
             data="nmv_visdrone_3cls.yaml", split="test", batch=4),
    ],
    # Same resolution, seeds and test protocol as the current 1280 headline table.
    "confirm": [
        dict(idx=26, base=BASE_NAME, scale_assign=True,
             suffix="_1280", seed=42, imgsz=1280,
             data="nmv_visdrone_3cls.yaml", split="test", batch=2),
        dict(idx=26, base=BASE_NAME, scale_assign=True,
             suffix="_1280_s1", seed=1, imgsz=1280,
             data="nmv_visdrone_3cls.yaml", split="test", batch=2),
        dict(idx=26, base=BASE_NAME, scale_assign=True,
             suffix="_1280_s7", seed=7, imgsz=1280,
             data="nmv_visdrone_3cls.yaml", split="test", batch=2),
    ],
    # Public-benchmark check on the official VisDrone 10-class validation set.
    "vis10": [
        # Include the two existing seed-42 controls for an idempotent complete
        # plan; train.py and the evaluator skip them when their outputs exist.
        dict(idx=1, base="E01_baseline", scale_assign=False,
             suffix="_vis10_960", seed=42, imgsz=960,
             data="visdrone10.yaml", split="val", batch=2),
        dict(idx=1, base="E01_baseline", scale_assign=False,
             suffix="_vis10_960_s1", seed=1, imgsz=960,
             data="visdrone10.yaml", split="val", batch=2),
        dict(idx=1, base="E01_baseline", scale_assign=False,
             suffix="_vis10_960_s7", seed=7, imgsz=960,
             data="visdrone10.yaml", split="val", batch=2),
        dict(idx=2, base="E02_p2", scale_assign=False,
             suffix="_vis10_960", seed=42, imgsz=960,
             data="visdrone10.yaml", split="val", batch=2),
        dict(idx=2, base="E02_p2", scale_assign=False,
             suffix="_vis10_960_s1", seed=1, imgsz=960,
             data="visdrone10.yaml", split="val", batch=2),
        dict(idx=2, base="E02_p2", scale_assign=False,
             suffix="_vis10_960_s7", seed=7, imgsz=960,
             data="visdrone10.yaml", split="val", batch=2),
        dict(idx=26, base=BASE_NAME, scale_assign=True,
             suffix="_vis10_960", seed=42, imgsz=960,
             data="visdrone10.yaml", split="val", batch=2),
        dict(idx=26, base=BASE_NAME, scale_assign=True,
             suffix="_vis10_960_s1", seed=1, imgsz=960,
             data="visdrone10.yaml", split="val", batch=2),
        dict(idx=26, base=BASE_NAME, scale_assign=True,
             suffix="_vis10_960_s7", seed=7, imgsz=960,
             data="visdrone10.yaml", split="val", batch=2),
    ],
}


def show_command(command, env_overrides):
    print("  environment:")
    for key in sorted(env_overrides):
        print(f"    {key}={env_overrides[key]}")
    print("  command:")
    print("    " + subprocess.list2cmdline([str(x) for x in command]))


def call(command, env=None, dry_run=False):
    show_command(command, env or {})
    if dry_run:
        return 0
    proc_env = os.environ.copy()
    if env:
        proc_env.update({key: str(value) for key, value in env.items()})
    return subprocess.run([str(x) for x in command], cwd=ROOT, env=proc_env).returncode


def run_one(spec, args):
    suffix = spec["suffix"]
    # Never let a short smoke run make the later 150-epoch formal run look
    # "completed". Non-standard epoch counts always receive a separate name.
    if args.epochs is not None and args.epochs != 150:
        suffix += f"_e{args.epochs}"
    run_name = spec["base"] + suffix
    data_yaml = ROOT / "configs" / "data" / spec["data"]
    audit_path = RUNS / run_name / "tal_assignment_stats.json"
    env = {
        "NMV_RUN_SUFFIX": suffix,
        "NMV_SEED": spec["seed"],
        "NMV_IMGSZ": spec["imgsz"],
        "NMV_BATCH": args.batch or spec["batch"],
        "NMV_DATA": data_yaml,
        "NMV_TAL_AUDIT": "1",
        "NMV_TAL_AUDIT_PATH": audit_path,
        "NMV_TAL_AUDIT_EVERY": args.audit_every,
        "NMV_TAL_AUDIT_TRAIN_ONLY": "1",
        "NMV_TAL_AUDIT_RESET": "1" if args.force else "0",
        "NMV_SCALE_ASSIGN": "1" if spec["scale_assign"] else "0",
        "NMV_SCALE_HI_RATIO": "32",
        "NMV_SCALE_LO_RATIO": "0",
        "NMV_IOU": "ciou",
        "NMV_NMS": "hard",
    }
    if args.epochs is not None:
        env["NMV_EPOCHS"] = args.epochs

    print("\n" + "=" * 88)
    print(f"TRAIN {run_name} | seed={spec['seed']} imgsz={spec['imgsz']} data={spec['data']}")
    print("=" * 88)
    train_cmd = [PYTHON, TRAIN, "--only", str(spec["idx"])]
    if args.force:
        train_cmd.append("--force")
    rc = call(train_cmd, env=env, dry_run=args.dry_run)
    if rc:
        print(f"[ERROR] training failed with exit code {rc}; evaluation was not started")
        return rc
    if args.skip_eval:
        return 0

    eval_batch = args.batch or spec["batch"]
    print(f"\nEVALUATE {run_name} on {spec['split']}")
    val_output = RUNS / f"{run_name}_eval_{spec['split']}" / "results.csv"
    val_cmd = [
        PYTHON, VAL,
        "--run", run_name,
        "--data", data_yaml,
        "--split", spec["split"],
        "--imgsz", str(spec["imgsz"]),
        "--batch", str(eval_batch),
    ]
    if val_output.exists() and not args.force:
        print(f"  [skip] standard evaluation exists: {val_output}")
    else:
        rc = call(val_cmd, dry_run=args.dry_run)
        if rc:
            return rc
    bucket_cmd = [
        PYTHON, BUCKETS,
        "--run", run_name,
        "--data", data_yaml,
        "--split", spec["split"],
        "--imgsz", str(spec["imgsz"]),
    ]
    bucket_output = RUNS / f"{run_name}_eval_{spec['split']}_buckets" / "metrics.json"
    if bucket_output.exists() and not args.force:
        print(f"  [skip] size-bucket evaluation exists: {bucket_output}")
        return 0
    return call(bucket_cmd, dry_run=args.dry_run)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the isolated P2+SA ablation (E26)."
    )
    parser.add_argument(
        "--stage", choices=["screen", "confirm", "vis10", "all"], default="screen",
        help="screen=one 960 run; confirm=three 1280 seeds; vis10=three official-val seeds",
    )
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override 150 epochs")
    parser.add_argument("--audit-every", type=int, default=100,
                        help="Write TAL audit JSON every N batches")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if this exact run directory is complete")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    selected = ["screen", "confirm", "vis10"] if args.stage == "all" else [args.stage]
    jobs = [job for stage in selected for job in STAGES[stage]]
    print(f"P2+SA supplement: stage={args.stage}, jobs={len(jobs)}, started={datetime.now()}")
    print("Existing completed runs are skipped; interrupted runs resume from last.pt.")
    for job in jobs:
        rc = run_one(job, args)
        if rc:
            sys.exit(rc)
    print(f"\n[DONE] {len(jobs)} job(s) finished at {datetime.now()}")


if __name__ == "__main__":
    main()
