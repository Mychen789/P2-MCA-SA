"""SAHI (Slicing Aided Hyper Inference) experiment for E07.

Reference: Akyon et al., "Slicing Aided Hyper Inference and Fine-tuning for Small
Object Detection", ICIP 2022 (arXiv:2202.06934).

Pipeline:
  1. Convert val/test labels (YOLO format) to COCO ground-truth JSON
  2. For each image, run SAHI sliced prediction with the trained model
  3. Save predictions in COCO format
  4. Use pycocotools to compute mAP@0.5 and mAP@0.5:0.95
  5. Print + save metrics

Usage:
  python scripts/sahi_predict.py --run E04_p2_ema_mpdiou_gfpn --split val
  python scripts/sahi_predict.py --run E04_p2_ema_mpdiou_gfpn --split test --slice 640 --overlap 0.25
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nmv  # noqa: F401  # triggers patch injection (not needed for SAHI but harmless)
from nmv.utils.data import find_data_root, CLASSES


def yolo_to_coco_gt(image_dir, label_dir, classes):
    """Convert YOLO-style labels to a COCO ground-truth dict.

    Returns the dict (caller saves to JSON). Image IDs are 1..N in sorted order.
    """
    img_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    images, annotations = [], []
    ann_id = 1
    for img_id, p in enumerate(img_paths, start=1):
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        images.append(dict(id=img_id, file_name=p.name, width=int(w), height=int(h)))
        lbl = label_dir / (p.stem + ".txt")
        if not lbl.exists():
            continue
        for ln in lbl.read_text().splitlines():
            parts = ln.split()
            if len(parts) < 5:
                continue
            c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            bw_abs = bw * w
            bh_abs = bh * h
            annotations.append(dict(
                id=ann_id, image_id=img_id, category_id=c + 1,
                bbox=[float(x1), float(y1), float(bw_abs), float(bh_abs)],
                area=float(bw_abs * bh_abs), iscrowd=0,
            ))
            ann_id += 1
    categories = [dict(id=i + 1, name=name) for i, name in enumerate(classes)]
    return dict(images=images, annotations=annotations, categories=categories)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Run dir name under runs/")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--slice", type=int, default=640, help="Slice size in pixels")
    p.add_argument("--overlap", type=float, default=0.2, help="Slice overlap ratio (0-1)")
    p.add_argument("--conf", type=float, default=0.01,
                   help="COCOeval 对 PR 曲线积分,conf 太高(如 0.25)会截断预测、压低 mAP(尤其 mAP_small)数 pp。"
                        "评估用 0.01(同 COCO 低阈值惯例);只在做可视化时才调高。")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    bp = ROOT / "runs" / args.run / "weights" / "best.pt"
    if not bp.exists():
        print(f"[ERROR] best.pt not found: {bp}")
        sys.exit(1)

    data_root = find_data_root()
    img_dir = data_root / "images" / args.split
    lbl_dir = data_root / "labels" / args.split
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[ERROR] split paths missing: {img_dir} or {lbl_dir}")
        sys.exit(1)

    out_dir = ROOT / "runs" / f"E07_sahi_{args.run}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== SAHI sliced inference for {args.run} on {args.split} split ===")
    print(f"  slice={args.slice}px  overlap={args.overlap}  conf={args.conf}")

    print("\n[1/3] Building COCO ground-truth from YOLO labels...")
    gt = yolo_to_coco_gt(img_dir, lbl_dir, CLASSES)
    gt_path = out_dir / "gt_coco.json"
    gt_path.write_text(json.dumps(gt))
    print(f"  {len(gt['images'])} images, {len(gt['annotations'])} GT boxes -> {gt_path}")

    print("\n[2/3] Running SAHI sliced prediction (this may take a while)...")
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from tqdm import tqdm

    det_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(bp),
        confidence_threshold=args.conf,
        device=args.device,
    )

    img_id_map = {im["file_name"]: im["id"] for im in gt["images"]}
    predictions = []
    for img_path in tqdm(sorted(img_dir.iterdir())):
        if img_path.suffix.lower() not in (".jpg", ".png", ".jpeg"):
            continue
        if img_path.name not in img_id_map:
            continue
        result = get_sliced_prediction(
            str(img_path),
            det_model,
            slice_height=args.slice,
            slice_width=args.slice,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            verbose=0,
        )
        for op in result.object_prediction_list:
            x1, y1, x2, y2 = op.bbox.to_xyxy()
            predictions.append(dict(
                image_id=img_id_map[img_path.name],
                category_id=op.category.id + 1,
                bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                score=float(op.score.value),
            ))

    pred_path = out_dir / "predictions_coco.json"
    pred_path.write_text(json.dumps(predictions))
    print(f"  {len(predictions)} predictions -> {pred_path}")

    print("\n[3/3] COCO evaluation...")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    metrics = dict(
        mAP_50_95=float(ev.stats[0]),
        mAP_50=float(ev.stats[1]),
        mAP_75=float(ev.stats[2]),
        mAP_small=float(ev.stats[3]),
        mAP_medium=float(ev.stats[4]),
        mAP_large=float(ev.stats[5]),
        AR_max1=float(ev.stats[6]),
        AR_max10=float(ev.stats[7]),
        AR_max100=float(ev.stats[8]),
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[OK] metrics -> {out_dir / 'metrics.json'}")
    print(f"  mAP@0.5      = {metrics['mAP_50']:.4f}")
    print(f"  mAP@0.5:0.95 = {metrics['mAP_50_95']:.4f}")
    print(f"  mAP small    = {metrics['mAP_small']:.4f}  <- the number SAHI is built for")


if __name__ == "__main__":
    main()
