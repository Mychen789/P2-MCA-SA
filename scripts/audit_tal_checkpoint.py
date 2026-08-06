"""Audit final TAL positive assignments from an existing checkpoint.

The script performs one diagnostic training epoch with an effectively zero
learning rate. It therefore exercises the real training loss and post-top-k
TaskAlignedAssigner using the checkpoint's learned predictions without doing a
full retraining. Assignment counts are written by scale_aware_assign.py.

Examples:
  # Existing P2 checkpoint, ordinary TAL
  python scripts/audit_tal_checkpoint.py --run E02_p2 --imgsz 960 --batch 4

  # Newly trained P2+SA checkpoint, scale-aware TAL
  python scripts/audit_tal_checkpoint.py \
      --run E26_p2_scale_assign_hi32_screen960 --imgsz 960 --batch 4 \
      --scale-assign --hi-ratio 32
"""
import argparse
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-epoch, near-zero-LR audit of post-top-k TAL assignments."
    )
    parser.add_argument("--run", required=True, help="Source run directory under runs/")
    parser.add_argument(
        "--data", default=str(ROOT / "configs" / "data" / "nmv_visdrone_3cls.yaml")
    )
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of the training set to audit (default: full set)")
    parser.add_argument("--scale-assign", action="store_true")
    parser.add_argument("--hi-ratio", type=float, default=32.0)
    parser.add_argument("--lo-ratio", type=float, default=0.0)
    parser.add_argument("--audit-every", type=int, default=50)
    parser.add_argument("--tag", default=None,
                        help="Optional output tag; defaults to tal or sa_hi<ratio>")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = RUNS / args.run / "weights" / "best.pt"
    if not checkpoint.exists():
        print(f"[ERROR] checkpoint not found: {checkpoint}")
        sys.exit(2)
    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"[ERROR] dataset YAML not found: {data_yaml}")
        sys.exit(2)

    ratio_tag = str(args.hi_ratio).replace(".", "p")
    tag = args.tag or (f"sa_hi{ratio_tag}" if args.scale_assign else "tal")
    job_name = f"{args.run}_{tag}_i{args.imgsz}_s{args.seed}"
    job_dir = RUNS / "_tal_audit_jobs" / job_name
    stats_path = job_dir / "tal_assignment_stats.json"
    audit_started_ns = time.time_ns()

    # These variables must be set before importing nmv/ultralytics because the
    # monkey patches are installed at import time.
    os.environ["NMV_TAL_AUDIT"] = "1"
    os.environ["NMV_TAL_AUDIT_PATH"] = str(stats_path)
    os.environ["NMV_TAL_AUDIT_EVERY"] = str(args.audit_every)
    os.environ["NMV_TAL_AUDIT_TRAIN_ONLY"] = "1"
    os.environ["NMV_TAL_AUDIT_RESET"] = "1"
    os.environ["NMV_SCALE_ASSIGN"] = "1" if args.scale_assign else "0"
    os.environ["NMV_SCALE_HI_RATIO"] = str(args.hi_ratio)
    os.environ["NMV_SCALE_LO_RATIO"] = str(args.lo_ratio)

    sys.path.insert(0, str(ROOT))
    import nmv  # noqa: F401,E402  installs TAL audit before YOLO creates its loss
    from nmv.patches.scale_aware_assign import flush_audit  # noqa: E402
    from ultralytics import YOLO  # noqa: E402

    mode = f"SA(hi={args.hi_ratio:g})" if args.scale_assign else "ordinary TAL"
    print(f"=== TAL checkpoint audit: {args.run} | {mode} ===")
    print(f"  checkpoint: {checkpoint}")
    print(f"  data:       {data_yaml}")
    print(f"  imgsz/batch/seed: {args.imgsz}/{args.batch}/{args.seed}")
    print(f"  dataset fraction: {args.fraction}")
    print(f"  output:     {stats_path}")
    print("  diagnostic epoch uses lr=1e-12 and zero weight decay")
    print("  validation-loss calls are excluded from assignment statistics")

    model = YOLO(str(checkpoint))
    model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        cache="disk",
        device=0,
        amp=True,
        seed=args.seed,
        deterministic=True,
        fraction=args.fraction,
        project=str(RUNS / "_tal_audit_jobs"),
        name=job_name,
        exist_ok=True,
        optimizer="SGD",
        lr0=1e-12,
        lrf=1.0,
        momentum=0.937,
        weight_decay=0.0,
        warmup_epochs=0.0,
        warmup_momentum=0.937,
        warmup_bias_lr=0.0,
        cos_lr=False,
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
        close_mosaic=0,
        val=False,
        save=False,
        plots=False,
        verbose=True,
    )
    flush_audit()
    # Ultralytics may still create final best/last checkpoints even with
    # save=False. They are diagnostic by-products, not results, so remove only
    # these two files while preserving the JSON and logs.
    for filename in ("best.pt", "last.pt"):
        generated = job_dir / "weights" / filename
        if generated.exists():
            generated.unlink()
    weights_dir = job_dir / "weights"
    if weights_dir.exists() and not any(weights_dir.iterdir()):
        weights_dir.rmdir()
    if stats_path.exists() and stats_path.stat().st_mtime_ns >= audit_started_ns:
        print(f"\n[OK] TAL audit saved: {stats_path}")
    else:
        print("\n[ERROR] diagnostic epoch finished but no fresh TAL audit JSON was written")
        sys.exit(1)


if __name__ == "__main__":
    main()
