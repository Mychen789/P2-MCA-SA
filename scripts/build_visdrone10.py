"""Converter: raw VisDrone2019-DET annotations -> standard 10-class YOLO dataset.

Purpose (2026-06-10): the paper needs a standard 10-class VisDrone anchor so the
3-class NMV results have a comparable reference against published numbers
(SL-YOLO 46.9 mAP50 etc., all reported on the official 10-class val split).
Produces the dataset referenced by configs/data/visdrone10.yaml, trained at 960
via:  NMV_DATA=configs/data/visdrone10.yaml NMV_IMGSZ=960 NMV_RUN_SUFFIX=_vd10_960

VisDrone-DET annotation format (per box):
  <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>

Official categories: 0=ignored-regions, 1=pedestrian, 2=people, 3=bicycle,
4=car, 5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others.
Standard 10-class protocol: categories 1-10 -> YOLO ids 0-9; drop 0/11 and
score==0 (ignored) boxes. Same convention as the official ultralytics converter.

Usage:
  python scripts/build_visdrone10.py \
      --visdrone-root /path/to/VisDrone2019-DET \
      --output-root   datasets/visdrone10_yolo
"""
import argparse
import shutil
from pathlib import Path

from PIL import Image

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
}


def convert_split(src_root: Path, out_root: Path, split: str, src_name: str):
    src_annos = src_root / src_name / "annotations"
    src_imgs = src_root / src_name / "images"
    if not src_annos.is_dir() or not src_imgs.is_dir():
        raise SystemExit(f"Expected {src_annos} and {src_imgs} to exist.")

    out_imgs = out_root / "images" / split
    out_labs = out_root / "labels" / split
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_labs.mkdir(parents=True, exist_ok=True)

    n_imgs, n_boxes, n_ignored = 0, 0, 0
    for anno_file in sorted(src_annos.glob("*.txt")):
        stem = anno_file.stem
        img_src = src_imgs / f"{stem}.jpg"
        if not img_src.exists():
            img_src = src_imgs / f"{stem}.png"
        if not img_src.exists():
            print(f"  [warn] image not found for {stem}, skipped")
            continue

        with Image.open(img_src) as im:
            W, H = im.size

        yolo_lines = []
        with open(anno_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().rstrip(",").split(",")
                if len(parts) < 6:
                    continue
                x, y, w, h, score, cat = (
                    int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]),
                    int(parts[4]), int(parts[5]),
                )
                if score == 0 or not (1 <= cat <= 10):
                    n_ignored += 1
                    continue
                if w <= 0 or h <= 0:
                    continue
                # clamp to image bounds (a few VisDrone boxes overflow the frame)
                x2, y2 = min(x + w, W), min(y + h, H)
                x1, y1 = max(x, 0), max(y, 0)
                if x2 <= x1 or y2 <= y1:
                    continue
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                nw, nh = (x2 - x1) / W, (y2 - y1) / H
                yolo_lines.append(f"{cat - 1} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        dst_img = out_imgs / img_src.name
        if not dst_img.exists():
            shutil.copy2(img_src, dst_img)
        with open(out_labs / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        n_imgs += 1
        n_boxes += len(yolo_lines)

    print(f"  [{split}] images: {n_imgs}, boxes kept: {n_boxes}, ignored/other lines: {n_ignored}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visdrone-root", required=True,
                   help="Dir containing VisDrone2019-DET-train/ and VisDrone2019-DET-val/.")
    p.add_argument("--output-root", required=True,
                   help="Output root; creates images/{train,val} and labels/{train,val}.")
    args = p.parse_args()

    src_root = Path(args.visdrone_root)
    out_root = Path(args.output_root)
    for split, src_name in SPLITS.items():
        print(f"converting {src_name} -> {split} ...")
        convert_split(src_root, out_root, split, src_name)
    print(f"\n=== Done === output: {out_root}")
    print("Next: train anchor runs with\n"
          "  NMV_DATA=configs/data/visdrone10.yaml NMV_IMGSZ=960 NMV_RUN_SUFFIX=_vd10_960")


if __name__ == "__main__":
    main()
