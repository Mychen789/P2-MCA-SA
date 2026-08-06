"""Resolve dataset roots across different machines.

The main paper dataset is NMV-SOD-3cls, but the training queue also needs a
true 10-class VisDrone anchor. Keep those YAMLs explicit so a requested 10-class
run cannot be silently rewritten into the 3-class dataset.
"""
import os
from pathlib import Path

CLASSES = ["ebike", "bicycle", "etrike"]
VISDRONE10_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def find_data_root():
    """Locate the nmv_visdrone_3cls dataset on this machine.

    Resolution order:
      1. NMV_DATA_ROOT environment variable
      2. <repo>/datasets/nmv_visdrone_3cls/ (inside the project repo)
      3. <repo>/../datasets/nmv_visdrone_3cls/ (alongside the project repo)
    """
    if os.environ.get("NMV_DATA_ROOT"):
        p = Path(os.environ["NMV_DATA_ROOT"])
        if (p / "images" / "train").exists():
            return p
        raise FileNotFoundError(
            f"NMV_DATA_ROOT={p} does not contain images/train"
        )

    here = Path(__file__).resolve().parents[2]
    candidates = [
        here / "datasets" / "nmv_visdrone_3cls",
        here.parent / "datasets" / "nmv_visdrone_3cls",
    ]
    for c in candidates:
        if (c / "images" / "train").exists():
            return c

    raise FileNotFoundError(
        "Dataset root not found. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nSet NMV_DATA_ROOT environment variable to your nmv_visdrone_3cls location."
    )


def find_visdrone10_root():
    """Locate the converted 10-class VisDrone YOLO dataset."""
    if os.environ.get("VISDRONE10_DATA_ROOT"):
        p = Path(os.environ["VISDRONE10_DATA_ROOT"])
        if (p / "images" / "train").exists() and (p / "images" / "val").exists():
            return p
        raise FileNotFoundError(
            f"VISDRONE10_DATA_ROOT={p} does not contain images/train and images/val"
        )

    here = Path(__file__).resolve().parents[2]
    candidates = [
        here / "datasets" / "raw" / "visdrone10_yolo",
        here.parent / "datasets" / "raw" / "visdrone10_yolo",
    ]
    for c in candidates:
        if (c / "images" / "train").exists() and (c / "images" / "val").exists():
            return c

    raise FileNotFoundError(
        "10-class VisDrone root not found. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nSet VISDRONE10_DATA_ROOT to the converted VisDrone YOLO dataset."
    )


def _write_yolo_data_yaml(yaml_path, root, classes, include_test=True):
    body = (
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
    )
    if include_test:
        body += "test: images/test\n"
    body += (
        "\n"
        f"nc: {len(classes)}\n"
        "names:\n"
    )
    for i, n in enumerate(classes):
        body += f"  {i}: {n}\n"
    if yaml_path.exists() and yaml_path.read_text(encoding="utf-8") == body:
        return yaml_path
    yaml_path.write_text(body, encoding="utf-8")
    return yaml_path


def ensure_data_yaml(yaml_path):
    """Return a usable data YAML for known datasets.

    Known generated files:
      - nmv_visdrone_3cls.yaml -> NMV-SOD-3cls, 3 classes, train/val/test
      - visdrone10.yaml        -> true 10-class VisDrone, train/val

    Existing unknown YAMLs are returned untouched to avoid accidental rewrites.
    """
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    if yaml_path.name == "nmv_visdrone_3cls.yaml":
        return _write_yolo_data_yaml(yaml_path, find_data_root(), CLASSES, include_test=True)

    if yaml_path.name == "visdrone10.yaml":
        return _write_yolo_data_yaml(
            yaml_path, find_visdrone10_root(), VISDRONE10_CLASSES, include_test=False
        )

    if yaml_path.exists():
        return yaml_path

    raise FileNotFoundError(f"Unknown data yaml and file does not exist: {yaml_path}")
