"""Qualitative detection visualization for the paper.

Two output modes:
  --mode pairs    GT | <each-experiment> side-by-side, one folder per experiment
  --mode compare  GT | E01 baseline | E11 full   in a single image (paper-ready)

Defaults to --mode compare on the test split, which is what the paper needs.
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nmv
from nmv.utils.data import find_data_root
from ultralytics import YOLO


CLASSES = ["ebike", "bicycle", "etrike"]
COLORS = [(255, 80, 80), (80, 200, 80), (80, 120, 255)]


def draw_boxes(img, boxes, labels, color_per_label, title, scores=None):
    out = img.copy()
    for i, ((x1, y1, x2, y2), lab) in enumerate(zip(boxes, labels)):
        c = color_per_label[int(lab) % len(color_per_label)]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        text = CLASSES[int(lab)] if 0 <= int(lab) < len(CLASSES) else str(lab)
        if scores is not None:
            text = f"{text} {scores[i]:.2f}"
        cv2.putText(out, text, (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (max(220, len(title) * 14), 36), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def load_gt(label_path, w, h):
    boxes, labels = [], []
    if not label_path.exists():
        return boxes, labels
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append((x1, y1, x2, y2))
        labels.append(c)
    return boxes, labels


def predict(model, img_path, conf=0.25):
    r = model.predict(str(img_path), imgsz=960, conf=conf, verbose=False)[0]
    return (
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.cls.cpu().numpy(),
        r.boxes.conf.cpu().numpy(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="compare", choices=["pairs", "compare"])
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--n-images", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--baseline-run", default="E01_baseline")
    p.add_argument("--full-run", default="E11_full_mca_cagfpn")
    args = p.parse_args()

    runs = ROOT / "runs"
    out_dir = runs / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = find_data_root()
    img_dir = data_root / "images" / args.split
    lbl_dir = data_root / "labels" / args.split
    if not img_dir.exists():
        print("Image dir not found:", img_dir)
        return

    all_imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    random.seed(args.seed)
    chosen = random.sample(all_imgs, min(args.n_images, len(all_imgs)))

    if args.mode == "compare":
        bp_base = runs / args.baseline_run / "weights" / "best.pt"
        bp_full = runs / args.full_run / "weights" / "best.pt"
        if not bp_base.exists() or not bp_full.exists():
            print(f"Missing weights: {bp_base} or {bp_full}")
            return
        print(f"=== compare {args.baseline_run} vs {args.full_run} on {args.split} ===")
        m_base = YOLO(str(bp_base))
        m_full = YOLO(str(bp_full))
        cmp_out = out_dir / f"vis_compare_{args.split}"
        cmp_out.mkdir(parents=True, exist_ok=True)
        for img_path in chosen:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            gt_boxes, gt_labels = load_gt(lbl_dir / (img_path.stem + ".txt"), w, h)
            pb, pl, ps = predict(m_base, img_path, conf=args.conf)
            fb, fl, fs = predict(m_full, img_path, conf=args.conf)
            left = draw_boxes(img, gt_boxes, gt_labels, COLORS, "Ground Truth")
            mid = draw_boxes(img, pb, pl, COLORS, f"Baseline ({args.baseline_run})", scores=ps)
            right = draw_boxes(img, fb, fl, COLORS, f"Full ({args.full_run})", scores=fs)
            cat = np.concatenate([left, mid, right], axis=1)
            cv2.imwrite(str(cmp_out / img_path.name), cat)
        print(f"  saved {len(chosen)} triplets -> {cmp_out}")
        return

    # mode == pairs
    exps_with_weights = []
    for d in sorted(runs.glob("E*")):
        if "_eval_test" in d.name or d.name.startswith("E07_sahi"):
            continue
        bp = d / "weights" / "best.pt"
        if bp.exists():
            exps_with_weights.append((d.name, bp))
    if not exps_with_weights:
        print("No experiments with best.pt found.")
        return
    for exp_name, bp in exps_with_weights:
        print(f"=== {exp_name} ===")
        model = YOLO(str(bp))
        exp_out = out_dir / f"vis_{exp_name}_{args.split}"
        exp_out.mkdir(parents=True, exist_ok=True)
        for img_path in chosen:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            gt_boxes, gt_labels = load_gt(lbl_dir / (img_path.stem + ".txt"), w, h)
            pb, pl, ps = predict(model, img_path, conf=args.conf)
            left = draw_boxes(img, gt_boxes, gt_labels, COLORS, "Ground Truth")
            right = draw_boxes(img, pb, pl, COLORS, exp_name, scores=ps)
            cat = np.concatenate([left, right], axis=1)
            cv2.imwrite(str(exp_out / img_path.name), cat)
        print(f"  saved {len(chosen)} pairs -> {exp_out}")


if __name__ == "__main__":
    main()
