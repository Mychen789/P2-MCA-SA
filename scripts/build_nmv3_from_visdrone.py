"""Rebuild the exact three-class manuscript dataset from official VisDrone.

The split manifests are repository metadata only; official images and
annotations must be downloaded separately. Output images keep a source-package
prefix so train and validation sequence identifiers remain unambiguous.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits"
VISDRONE_TO_NMV = {10: 0, 3: 1, 7: 2, 8: 2}


def source_for(stem: str, train_root: Path, val_root: Path) -> tuple[Path, str]:
    if stem.startswith("visdrone_train_"):
        return train_root, stem.removeprefix("visdrone_train_")
    if stem.startswith("visdrone_val_"):
        return val_root, stem.removeprefix("visdrone_val_")
    raise ValueError(f"Unrecognised source prefix in split stem: {stem}")


def find_image(images: Path, raw_stem: str) -> Path:
    for suffix in (".jpg", ".png", ".jpeg"):
        candidate = images / f"{raw_stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for {raw_stem} under {images}")


def convert_annotation(annotation: Path, width: int, height: int) -> list[str]:
    converted: list[str] = []
    for line in annotation.read_text(encoding="utf-8").splitlines():
        parts = line.rstrip(",").split(",")
        if len(parts) < 6:
            continue
        x, y, box_w, box_h, score, category = map(int, parts[:6])
        if score == 0 or category not in VISDRONE_TO_NMV or box_w <= 0 or box_h <= 0:
            continue
        cx = (x + box_w / 2) / width
        cy = (y + box_h / 2) / height
        norm_w = box_w / width
        norm_h = box_h / height
        converted.append(
            f"{VISDRONE_TO_NMV[category]} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}"
        )
    return converted


def place_image(source: Path, destination: Path, hardlink: bool) -> None:
    if hardlink:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def build_split(
    split: str, train_root: Path, val_root: Path, output: Path, hardlink: bool
) -> tuple[int, int]:
    stems = [
        line.strip().lstrip("\ufeff")
        for line in (SPLITS / f"{split}.txt").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    image_output = output / "images" / split
    label_output = output / "labels" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    box_count = 0
    for stem in stems:
        source_root, raw_stem = source_for(stem, train_root, val_root)
        source_image = find_image(source_root / "images", raw_stem)
        source_annotation = source_root / "annotations" / f"{raw_stem}.txt"
        if not source_annotation.exists():
            raise FileNotFoundError(source_annotation)
        with Image.open(source_image) as image:
            width, height = image.size
        lines = convert_annotation(source_annotation, width, height)
        box_count += len(lines)
        place_image(source_image, image_output / f"{stem}{source_image.suffix.lower()}", hardlink)
        (label_output / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    return len(stems), box_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardlink", action="store_true")
    args = parser.parse_args()

    for root in (args.train_root, args.val_root):
        if not (root / "images").is_dir() or not (root / "annotations").is_dir():
            raise SystemExit(f"Expected images/ and annotations/ under {root}")

    for split in ("train", "val", "test"):
        images, boxes = build_split(
            split, args.train_root, args.val_root, args.output, args.hardlink
        )
        print(f"{split}: {images} images, {boxes} retained boxes")

    print(f"Dataset written to {args.output.resolve()}")


if __name__ == "__main__":
    main()

