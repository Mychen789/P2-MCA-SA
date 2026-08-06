"""Dataset bounding-box size & class statistics for the paper's dataset section.

For each split (train/val/test):
  - count images / labels
  - per-class count
  - bbox absolute pixel area (= bw * imgW * bh * imgH), split into
    COCO categories: small (<32^2), medium (32^2..96^2), large (>96^2)

Outputs to runs/_summary/:
  - dataset_stats.csv         headline counts
  - box_sizes_sqrt_area.csv   every box's sqrt(area) in pixels, long form
                              (split,sqrt_area_px) — the raw data behind Fig. 1's
                              continuous size-distribution panel; matches
                              dataset_stats.csv's small/medium/large fractions
                              exactly because it is the same area = bw*imgW*bh*imgH
                              computed here, only not yet bucketed.
  - dataset_size_hist.png     histogram of sqrt(area) in pixels, train split
  - dataset_class_dist.png    stacked bar of class counts per split
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nmv.utils.data import find_data_root, CLASSES


def parse_split(data_root, split):
    img_dir = data_root / "images" / split
    lbl_dir = data_root / "labels" / split
    sizes_sqrt = []
    cls_counter = Counter()
    coco_buckets = {"small": 0, "medium": 0, "large": 0}
    n_imgs = 0
    if not img_dir.exists():
        return None
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        n_imgs += 1
        lbl = lbl_dir / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                bw = float(parts[3]) * w
                bh = float(parts[4]) * h
            except ValueError:
                continue
            area = bw * bh
            sizes_sqrt.append(np.sqrt(area))
            cls_counter[cls] += 1
            if area < 32 ** 2:
                coco_buckets["small"] += 1
            elif area < 96 ** 2:
                coco_buckets["medium"] += 1
            else:
                coco_buckets["large"] += 1
    return dict(n_imgs=n_imgs, sizes_sqrt=sizes_sqrt,
                cls_counter=cls_counter, coco_buckets=coco_buckets)


def main():
    data_root = find_data_root()
    out_dir = ROOT / "runs" / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"data_root = {data_root}")
    stats = {}
    for split in ("train", "val", "test"):
        s = parse_split(data_root, split)
        if s is None:
            print(f"[skip] {split} not found")
            continue
        stats[split] = s
        total = sum(s["coco_buckets"].values())
        small_pct = 100 * s["coco_buckets"]["small"] / max(1, total)
        print(f"{split:5s}: {s['n_imgs']} imgs, {total} boxes, "
              f"small {s['coco_buckets']['small']} ({small_pct:.1f}%), "
              f"medium {s['coco_buckets']['medium']}, "
              f"large {s['coco_buckets']['large']}")

    csv = ["split,n_images,n_boxes,small,medium,large," + ",".join(CLASSES)]
    for split, s in stats.items():
        total = sum(s["coco_buckets"].values())
        per_cls = [s["cls_counter"].get(i, 0) for i in range(len(CLASSES))]
        csv.append(f"{split},{s['n_imgs']},{total},"
                   f"{s['coco_buckets']['small']},{s['coco_buckets']['medium']},{s['coco_buckets']['large']},"
                   + ",".join(str(c) for c in per_cls))
    (out_dir / "dataset_stats.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")

    raw = ["split,sqrt_area_px"]
    for split, s in stats.items():
        raw += [f"{split},{v:.3f}" for v in s["sizes_sqrt"]]
    (out_dir / "box_sizes_sqrt_area.csv").write_text("\n".join(raw) + "\n", encoding="utf-8")
    print(f"Saved -> {out_dir / 'box_sizes_sqrt_area.csv'} ({len(raw) - 1} boxes)")

    if "train" in stats and stats["train"]["sizes_sqrt"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sizes = np.array(stats["train"]["sizes_sqrt"])
        ax.hist(sizes, bins=60, range=(0, min(300, sizes.max())),
                color="#3a7bd5", edgecolor="white")
        ax.axvline(32, color="orange", linestyle="--", label="COCO small (32px)")
        ax.axvline(96, color="red", linestyle="--", label="COCO medium (96px)")
        ax.set_xlabel("sqrt(box area)  [pixels]")
        ax.set_ylabel("count")
        ax.set_title(f"Train-split bounding-box size distribution  (n={len(sizes)})")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "dataset_size_hist.png", dpi=150)
        print(f"Saved -> {out_dir / 'dataset_size_hist.png'}")

    splits = list(stats.keys())
    per_cls_mat = np.array([
        [stats[sp]["cls_counter"].get(i, 0) for i in range(len(CLASSES))]
        for sp in splits
    ])
    if per_cls_mat.size:
        fig, ax = plt.subplots(figsize=(7, 4))
        bottoms = np.zeros(len(splits))
        colors = ["#3a7bd5", "#48b56a", "#e6783f"]
        for i, c in enumerate(CLASSES):
            ax.bar(splits, per_cls_mat[:, i], bottom=bottoms,
                   label=c, color=colors[i % len(colors)])
            bottoms = bottoms + per_cls_mat[:, i]
        ax.set_ylabel("# boxes")
        ax.set_title("Per-split class distribution")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "dataset_class_dist.png", dpi=150)
        print(f"Saved -> {out_dir / 'dataset_class_dist.png'}")

    print(f"Saved -> {out_dir / 'dataset_stats.csv'}")


if __name__ == "__main__":
    main()
