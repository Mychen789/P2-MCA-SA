"""Recompute the SA candidate-eligibility table behind Fig. 2c and section 3.4.

Two distinct bases exist and the manuscript must not conflate them:

1. **Source resolution.** Boxes measured in the original image's pixels. This is
   the basis on which COCO defines small/medium/large, and therefore the basis
   of every AP_small / AP_medium / AP_large number in the paper. A dataset-level
   eligibility statistic on this basis is the one that is comparable to those AP
   results, which is why Fig. 2c uses it.

2. **Training input.** What the rule in `nmv/patches/scale_aware_assign.py`
   actually sees: Ultralytics letterboxes the long side to `imgsz`, so every box
   is scaled by `imgsz / max(W, H)` before `max(w, h) < rho_hi * stride` is
   evaluated. Because that shrinks boxes, far fewer objects are "large" at
   training time than at source resolution, and the realised exclusion rate
   differs. Reporting the source-resolution rate as if it described training-time
   behaviour is the error this script exists to make impossible.

The inequality is strict (`<`), matching `_scale_mask`'s `gsize < hi`. An earlier
hand-built CSV used `<=` and therefore reported 500/685 large boxes losing P2
where the implementation excludes 501.

Mosaic and the other train-time augmentations are NOT modelled here: they resize
and re-crop boxes per batch, so a static table cannot describe them. The
per-batch ground truth of what the rule did during training is the TAL audit in
`runs/_tal_audit_jobs/*/tal_assignment_stats.json`, not this file.

Usage:
    python scripts/scale_assign_eligibility.py
    python scripts/scale_assign_eligibility.py --split train --write-csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "configs" / "data" / "nmv_visdrone_3cls.yaml"
OUT_CSV = ROOT / "runs" / "_summary" / "scale_assignment_eligibility.csv"

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# Detection strides present once the P2 head is added. P5 is exempt from any
# upper bound so that every object keeps at least one usable level (section 3.4).
STRIDES = [4, 8, 16, 32]
HEAD_OF = {4: "P2", 8: "P3", 16: "P4", 32: "P5"}
RHO_HI = 32.0

# COCO area thresholds.
SMALL_MAX = 32 * 32
MEDIUM_MAX = 96 * 96


def load_boxes(split: str):
    """Return [(w_px, h_px, long_image_side)] for every box in `split`."""
    cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf8"))
    root = Path(cfg["path"])
    img_dir = root / cfg[split]
    lbl_dir = root / cfg[split].replace("images", "labels", 1)

    out = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        W, H = Image.open(img).size
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            out.append((float(parts[3]) * W, float(parts[4]) * H, max(W, H)))
    return out


def finest_eligible_head(long_side: float) -> str:
    """Coarsest-to-finest: the finest level whose upper bound admits this box.

    Mirrors `_scale_mask`: level with stride s admits a box iff
    max(w, h) < RHO_HI * s, with the largest stride exempt (bound = inf).
    """
    for s in STRIDES:
        bound = float("inf") if s == STRIDES[-1] else RHO_HI * s
        if long_side < bound:
            return HEAD_OF[s]
    return HEAD_OF[STRIDES[-1]]


def tabulate(boxes, imgsz: int | None):
    """Cross-tabulate COCO size bucket against finest eligible head.

    `imgsz=None` measures at source resolution; an int applies the letterbox
    scale `imgsz / max(W, H)` that the training pipeline applies before the rule
    is evaluated.
    """
    buckets = {b: {h: 0 for h in ("P2", "P3", "P4", "P5")} for b in ("small", "medium", "large")}
    for w, h, img_long in boxes:
        scale = 1.0 if imgsz is None else imgsz / img_long
        sw, sh = w * scale, h * scale
        area = sw * sh
        bucket = "small" if area < SMALL_MAX else ("medium" if area < MEDIUM_MAX else "large")
        buckets[bucket][finest_eligible_head(max(sw, sh))] += 1
    return buckets


def render(label, buckets):
    print(f"\n=== {label} ===")
    print(f"{'bucket':8} {'total':>7} {'P2-P5':>8} {'P3-P5':>7} {'P4-P5':>7} {'P5':>5}   lose P2")
    for b in ("small", "medium", "large"):
        r = buckets[b]
        total = sum(r.values())
        lost = total - r["P2"]
        pct = f"{100 * lost / total:.1f}%" if total else "—"
        print(f"{b:8} {total:7d} {r['P2']:8d} {r['P3']:7d} {r['P4']:7d} {r['P5']:5d}   {lost:4d} = {pct}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--write-csv", action="store_true",
                    help=f"overwrite {OUT_CSV.relative_to(ROOT)} with the source-resolution table")
    args = ap.parse_args()

    boxes = load_boxes(args.split)
    print(f"{args.split}: {len(boxes)} boxes, rho_hi={RHO_HI:g}, strict '<' (matches _scale_mask)")

    source = tabulate(boxes, None)
    render("source resolution  (basis of Fig. 2c and of the COCO AP buckets)", source)
    for imgsz in (960, 1280):
        render(f"training input imgsz={imgsz}  (basis the rule actually sees)",
               tabulate(boxes, imgsz))

    if args.write_csv:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with OUT_CSV.open("w", newline="", encoding="utf8") as fh:
            w = csv.writer(fh)
            w.writerow(["bucket", "total", "P2+P3+P4+P5", "P3+P4+P5", "P4+P5", "P5", "basis"])
            for b in ("small", "medium", "large"):
                r = source[b]
                w.writerow([b, sum(r.values()), r["P2"], r["P3"], r["P4"], r["P5"], "source_resolution"])
        print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
