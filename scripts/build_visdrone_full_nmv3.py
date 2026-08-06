"""Converter: VisDrone-DET val annotations -> YOLO-format labels with the
NMV-SOD-3cls class id ordering (0=ebike, 1=bicycle, 2=etrike).

This produces the dataset referenced by configs/data/visdrone_full_nmv3.yaml,
used for the §5.7 cross-dataset zero-shot generalisation evaluation.

Differences vs. the original NMV-SOD-3cls construction:
  - NMV-SOD-3cls KEEPS only images containing at least one non-motorized vehicle.
  - This converter KEEPS ALL VisDrone val images, including ones with zero
    non-motorized boxes - the empty labels test the model's false-positive rate
    on broader urban scenes (cars, pedestrians, trucks, etc.).
  - Output is restricted to images that are NOT in the existing NMV-SOD-3cls
    train/val/test split (a true held-out set).

VisDrone-DET annotation format (per box):
  <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>

Class id mapping (VisDrone official -> NMV-SOD-3cls target):
  3  (bicycle)         -> 1 (bicycle)
  10 (motor)           -> 0 (ebike, approximate by geometry)
  7  (tricycle)        -> 2 (etrike)
  8  (awning-tricycle) -> 2 (etrike, approximate)
  All others           -> dropped
(2026-06-10 修正: 旧版误写 6->0, 但 VisDrone 官方类别 6 是 truck、motor 是 10。
 已用 nmv_visdrone_3cls 主数据集逐框反查确认 10->0 才与主数据集一致。)

Usage:
  python scripts/build_visdrone_full_nmv3.py \
      --visdrone-root  D:/datasets/VisDrone2019-DET-val \
      --nmv-test-list  D:/nmv_visdrone_3cls/test_images.txt \
      --output-root    D:/visdrone_full_nmv3

After this script completes, run:
  python scripts/eval_cross_dataset.py --run E17_p2_mca_p5trans_scale \
      --data configs/data/visdrone_full_nmv3.yaml --name visdrone_full
"""
import argparse
import shutil
from pathlib import Path

from PIL import Image


VISDRONE_TO_NMV = {
    3: 1,    # bicycle -> bicycle
    10: 0,   # motor -> ebike  (官方类别 10; 旧版误写 6=truck, 已修)
    7: 2,    # tricycle -> etrike
    8: 2,    # awning-tricycle -> etrike
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visdrone-root", required=True,
                   help="Path to VisDrone2019-DET-val with annotations/ and images/ subdirs.")
    p.add_argument("--nmv-test-list", default=None,
                   help="(Optional) Newline-separated list of image stems already in NMV-SOD-3cls; "
                        "if set, those images are EXCLUDED so the held-out set is truly unseen.")
    p.add_argument("--output-root", required=True,
                   help="Output root, will create images/val and labels/val inside.")
    args = p.parse_args()

    src_root = Path(args.visdrone_root)
    src_annos = src_root / "annotations"
    src_imgs = src_root / "images"
    if not src_annos.is_dir() or not src_imgs.is_dir():
        raise SystemExit(f"Expected {src_annos} and {src_imgs} to exist.")

    out_root = Path(args.output_root)
    out_imgs = out_root / "images" / "val"
    out_labs = out_root / "labels" / "val"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_labs.mkdir(parents=True, exist_ok=True)

    excluded = set()
    if args.nmv_test_list:
        with open(args.nmv_test_list, encoding="utf-8") as f:
            excluded = {Path(line.strip()).stem for line in f if line.strip()}
        print(f"  loaded {len(excluded)} excluded stems from {args.nmv_test_list}")

    n_imgs, n_kept, n_dropped, n_boxes_kept = 0, 0, 0, 0
    for anno_file in sorted(src_annos.glob("*.txt")):
        stem = anno_file.stem
        n_imgs += 1
        if stem in excluded:
            n_dropped += 1
            continue

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
                if score == 0:  # VisDrone DET: score=0 表示 ignored 标注
                    continue
                if cat not in VISDRONE_TO_NMV:
                    continue
                if w <= 0 or h <= 0:
                    continue
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                nw, nh = w / W, h / H
                if not (0 < nw <= 1 and 0 < nh <= 1):
                    continue
                yolo_lines.append(
                    f"{VISDRONE_TO_NMV[cat]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                )

        # Keep image even if no boxes - tests FP rate on scenes with no NMV objects.
        shutil.copy2(img_src, out_imgs / img_src.name)
        with open(out_labs / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        n_kept += 1
        n_boxes_kept += len(yolo_lines)

    print(f"\n=== Done ===")
    print(f"  total annotation files scanned: {n_imgs}")
    print(f"  excluded (in NMV split):        {n_dropped}")
    print(f"  kept:                            {n_kept}")
    print(f"  total NMV-class boxes kept:      {n_boxes_kept}")
    print(f"  output: {out_root}")


if __name__ == "__main__":
    main()
