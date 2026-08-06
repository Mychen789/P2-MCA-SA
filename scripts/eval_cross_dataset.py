"""Cross-dataset zero-shot evaluation of NMV-trained models.

For Neurocomputing §5.7 (cross-dataset generalisation). Loads a best.pt trained
on NMV-SOD-3cls and runs val on a different dataset's data.yaml WITHOUT any
re-training, then writes the mAP into _summary/cross_dataset_eval.csv for the
paper table.

Two target datasets we plan to evaluate:
  (1) VisDrone-full: VisDrone-DET val split, filtered to the 3 non-motorized
      classes present in NMV-SOD-3cls (bicycle, motor->ebike, tricycle->etrike)
      but on images that were NOT in the train/val/test split of NMV-SOD-3cls.
      This is a "broader scene" generalisation test - the model sees the same
      class definitions but on images containing many other classes.
  (2) UAVDT (if class mapping can be reasonably constructed): UAVDT only has
      vehicle (car/bus/truck) classes natively, so a direct mapping does not
      exist. If needed we can hand-label or skip UAVDT and rely on VisDrone-full
      alone.

Usage:
  python scripts/eval_cross_dataset.py --run E17_p2_mca_p5trans_scale \
      --data configs/data/visdrone_full_nmv3.yaml \
      --name visdrone_full
  python scripts/eval_cross_dataset.py --run E16_mca_scale_assign_hi32 \
      --data configs/data/uavdt_nmv3.yaml --name uavdt

Notes:
  - The data.yaml must use the same class id ordering as nmv_visdrone_3cls.yaml
    (0=ebike, 1=bicycle, 2=etrike) or the mAP numbers will be meaningless.
  - imgsz=960 + iou=0.7 matches the NMV training/eval setting; same as scripts/val.py.
"""
import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nmv  # noqa: F401 — triggers patch injection

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True,
                   help="Run dir under runs/ whose weights/best.pt to load (e.g. E17_p2_mca_p5trans_scale).")
    p.add_argument("--data", required=True,
                   help="Path to the cross-dataset data.yaml (must use NMV-3cls class ids).")
    p.add_argument("--name", required=True,
                   help="Short identifier of the target dataset (e.g. 'visdrone_full', 'uavdt').")
    p.add_argument("--split", default="val",
                   help="Which split key in the data.yaml to evaluate (default 'val').")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--conf", type=float, default=0.001)
    args = p.parse_args()

    run_dir = ROOT / "runs" / args.run
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[ERROR] {best_pt} does not exist. Run the producing training first.")
        sys.exit(1)
    data_yaml = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    if not data_yaml.exists():
        print(f"[ERROR] {data_yaml} does not exist.")
        sys.exit(1)

    eval_name = f"{args.run}__cross_{args.name}"
    out_dir = ROOT / "runs" / eval_name

    print(f"=== Cross-dataset zero-shot evaluation ===")
    print(f"  source weights:   {best_pt}")
    print(f"  target dataset:   {args.name}  ({data_yaml})")
    print(f"  split:            {args.split}")
    print(f"  imgsz={args.imgsz} batch={args.batch} iou={args.iou} conf={args.conf}")
    print(f"  output run dir:   {out_dir}")

    t0 = datetime.now()
    model = YOLO(str(best_pt))
    results = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        iou=args.iou,
        conf=args.conf,
        workers=0,
        project=str(ROOT / "runs"),
        name=eval_name,
        exist_ok=True,
        verbose=True,
        plots=True,
    )
    elapsed = datetime.now() - t0
    print(f"\n  elapsed: {elapsed}")

    map50 = float(results.box.map50)
    map5095 = float(results.box.map)
    p_overall = float(results.box.mp)
    r_overall = float(results.box.mr)

    print(f"  mAP@0.5      = {map50:.4f}")
    print(f"  mAP@0.5:0.95 = {map5095:.4f}")
    print(f"  P            = {p_overall:.4f}")
    print(f"  R            = {r_overall:.4f}")

    # Append to the paper-asset summary CSV.
    summary_csv = ROOT / "runs" / "_summary" / "cross_dataset_eval.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["source_run", "target_dataset", "split", "P", "R", "mAP50", "mAP50_95"])
        w.writerow([args.run, args.name, args.split,
                    f"{p_overall:.4f}", f"{r_overall:.4f}",
                    f"{map50:.4f}", f"{map5095:.4f}"])
    print(f"  -> appended to {summary_csv}")


if __name__ == "__main__":
    main()
