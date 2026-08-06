"""COCO-style size-bucket mAP (small / medium / large) on any trained run.

Why: ultralytics val.py 默认只给整体 mAP；论文"小目标专项"叙事需要 mAP_small
量化证据。本脚本走 pycocotools，输出和 sahi_predict.py 同样的指标但不切片。

定义 (COCO):
  small  : area < 32^2 = 1024 px^2
  medium : 32^2 <= area < 96^2
  large  : area >= 96^2

Usage:
  python scripts/eval_size_buckets.py --run E11_full_mca_cagfpn --split test
  python scripts/eval_size_buckets.py --run E08_p2_mca --split test --conf 0.001

Output:
  runs/<run>_eval_test_buckets/
    gt_coco.json
    predictions_coco.json
    metrics.json        含 mAP_small / mAP_medium / mAP_large / mAP_50 / ...
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nmv  # noqa: F401
from nmv.utils.data import ensure_data_yaml


def load_dataset_yaml(path, split):
    yaml_path = ensure_data_yaml(Path(path))
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    if split not in cfg:
        raise KeyError(f"Split '{split}' not found in {yaml_path}")
    names = cfg.get("names", [])
    if isinstance(names, dict):
        classes = [names[i] for i in sorted(names, key=lambda x: int(x))]
    else:
        classes = list(names)
    return root, Path(cfg[split]), classes


def yolo_to_coco_gt(image_dir, label_dir, classes):
    img_paths = sorted([p for p in image_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
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
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--data", default=str(ROOT / "configs" / "data" / "nmv_visdrone_3cls.yaml"),
                   help="Dataset YAML. Defaults to NMV-SOD-3cls; use configs/data/visdrone10.yaml for official VisDrone val.")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.001,
                   help="低 conf 阈值才能让 pycocotools 的 PR 曲线完整 (默认 0.001 同 COCO 评估惯例)")
    p.add_argument("--iou", type=float, default=0.7,
                   help="NMS IoU (ultralytics 默认 0.7)")
    args = p.parse_args()

    bp = ROOT / "runs" / args.run / "weights" / "best.pt"
    if not bp.exists():
        print(f"[ERROR] best.pt not found: {bp}")
        sys.exit(1)

    data_root, split_rel, classes = load_dataset_yaml(args.data, args.split)
    img_dir = data_root / split_rel
    lbl_dir = data_root / str(split_rel).replace("images", "labels", 1)
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[ERROR] split paths missing: {img_dir} or {lbl_dir}")
        sys.exit(1)

    out_dir = ROOT / "runs" / f"{args.run}_eval_{args.split}_buckets"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Size-bucket eval: {args.run} on {args.split} ===")
    print(f"  imgsz={args.imgsz}  conf={args.conf}  iou={args.iou}")

    print("\n[1/3] Building COCO ground-truth from YOLO labels...")
    gt = yolo_to_coco_gt(img_dir, lbl_dir, classes)
    gt_path = out_dir / "gt_coco.json"
    gt_path.write_text(json.dumps(gt))
    print(f"  {len(gt['images'])} images, {len(gt['annotations'])} GT boxes")

    print("\n[2/3] Running standard (non-sliced) prediction...")
    from ultralytics import YOLO
    from tqdm import tqdm

    model = YOLO(str(bp))
    img_id_map = {im["file_name"]: im["id"] for im in gt["images"]}
    predictions = []
    for img_path in tqdm(sorted(img_dir.iterdir())):
        if img_path.suffix.lower() not in (".jpg", ".png", ".jpeg"):
            continue
        if img_path.name not in img_id_map:
            continue
        r = model.predict(
            str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        score = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, score):
            predictions.append(dict(
                image_id=img_id_map[img_path.name],
                category_id=int(c) + 1,
                bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                score=float(s),
            ))

    pred_path = out_dir / "predictions_coco.json"
    pred_path.write_text(json.dumps(predictions))
    print(f"  {len(predictions)} predictions")

    print("\n[3/3] COCO evaluation...")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()

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
        AR_small=float(ev.stats[9]),
        AR_medium=float(ev.stats[10]),
        AR_large=float(ev.stats[11]),
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[OK] -> {out_dir / 'metrics.json'}")
    print(f"  mAP@0.5:0.95 = {metrics['mAP_50_95']:.4f}")
    print(f"  mAP@0.5      = {metrics['mAP_50']:.4f}")
    print(f"  mAP small    = {metrics['mAP_small']:.4f}   <- 论文小目标关键数字")
    print(f"  mAP medium   = {metrics['mAP_medium']:.4f}")
    print(f"  mAP large    = {metrics['mAP_large']:.4f}")


if __name__ == "__main__":
    main()
