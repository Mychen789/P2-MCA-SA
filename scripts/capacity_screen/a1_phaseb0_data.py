from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import img2label_paths
from ultralytics.utils.patches import imread


SCHEMA = "A1_PHASEB0_JPEG_ONLY_DATA_V1"


class ReadOnlyJpegYOLODataset(YOLODataset):
    """YOLODataset with a fail-closed label cache and an unconditional JPEG decode path.

    The installed Ultralytics BaseDataset checks a sibling ``.npy`` even when
    ``cache=False``.  VisDrone-10 already contains such files, so Phase-B must
    bypass that implicit input source.  Labels are parsed directly from frozen
    TXT files; source ``*.cache`` files are neither read nor rebuilt.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        cache = kwargs.get("cache", False)
        if cache not in {False, None}:
            raise ValueError("ReadOnlyJpegYOLODataset requires cache=False/None")
        kwargs["cache"] = False
        super().__init__(*args, **kwargs)
        self.phaseb0_input_policy = SCHEMA

    def get_img_files(self, img_path: str | Path | list[str] | list[Path]) -> list[str]:
        """Accept a frozen direct-JPEG list without treating each JPEG as a text manifest."""
        if isinstance(img_path, (list, tuple)):
            if not img_path:
                raise FileNotFoundError("empty direct-JPEG manifest")
            resolved: list[str] = []
            seen: set[str] = set()
            for raw in img_path:
                path = Path(raw).resolve()
                identity = str(path).casefold()
                if identity in seen:
                    raise RuntimeError(f"duplicate direct-JPEG identity: {path}")
                if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
                    raise FileNotFoundError(f"missing or non-JPEG direct input: {path}")
                seen.add(identity)
                resolved.append(str(path))
            return resolved
        return super().get_img_files(img_path)

    def get_labels(self) -> list[dict[str, Any]]:
        self.label_files = img2label_paths(self.im_files)
        if not self.label_files:
            raise RuntimeError("no label files resolved")
        if any(Path(path).suffix.lower() not in {".jpg", ".jpeg"} for path in self.im_files):
            raise RuntimeError("Phase-B input policy accepts JPEG files only")
        labels: list[dict[str, Any]] = []
        duplicate_rows = 0
        raw_rows = 0
        for image_file, label_file in zip(self.im_files, self.label_files):
            image_path, label_path = Path(image_file), Path(label_file)
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            with Image.open(image_path) as image:
                if image.format != "JPEG":
                    raise RuntimeError(f"non-JPEG input: {image_path}")
                width, height = image.size
            physical = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if physical:
                if any(len(row) != 5 for row in physical):
                    raise RuntimeError(f"five-column detection labels required: {label_path}")
                matrix = np.asarray(physical, dtype=np.float32)
                if not np.isfinite(matrix).all():
                    raise RuntimeError(f"non-finite label: {label_path}")
                if (matrix[:, 0] != np.floor(matrix[:, 0])).any() or matrix[:, 0].min() < 0 or matrix[:, 0].max() >= 10:
                    raise RuntimeError(f"invalid class label: {label_path}")
                if matrix[:, 1:].min() < 0 or matrix[:, 1:].max() > 1 or (matrix[:, 3:5] <= 0).any():
                    raise RuntimeError(f"invalid normalized box: {label_path}")
                raw_rows += len(matrix)
                _, indices = np.unique(matrix, axis=0, return_index=True)
                duplicate_rows += len(matrix) - len(indices)
                matrix = matrix[indices]
            else:
                matrix = np.zeros((0, 5), dtype=np.float32)
            labels.append(
                {
                    "im_file": str(image_path),
                    "shape": (height, width),
                    "cls": matrix[:, 0:1],
                    "bboxes": matrix[:, 1:5],
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        self.phaseb0_raw_label_rows = raw_rows
        self.phaseb0_duplicate_rows_removed = duplicate_rows
        self.phaseb0_effective_label_rows = raw_rows - duplicate_rows
        self.phaseb0_label_source = "DIRECT_UTF8_TXT_FLOAT32_NP_UNIQUE"
        return labels

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Mirror Ultralytics 8.4.37 BaseDataset.load_image without any .npy branch."""
        im, file_name = self.ims[i], self.im_files[i]
        if im is None:
            im = imread(file_name, flags=self.cv2_flag)
            if im is None:
                raise FileNotFoundError(f"JPEG image not found or undecodable: {file_name}")
            h0, w0 = im.shape[:2]
            if rect_mode:
                ratio = self.imgsz / max(h0, w0)
                if ratio != 1:
                    width = min(math.ceil(w0 * ratio), self.imgsz)
                    height = min(math.ceil(h0 * ratio), self.imgsz)
                    im = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]

            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    old_index = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[old_index], self.im_hw0[old_index], self.im_hw[old_index] = None, None, None
            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]


def build_read_only_jpeg_dataset(
    *,
    cfg: Any,
    image_path: Path | None = None,
    image_paths: list[Path] | None = None,
    data: dict[str, Any],
    batch_size: int,
    augment: bool,
    rect: bool = False,
    stride: int = 32,
) -> ReadOnlyJpegYOLODataset:
    if (image_path is None) == (image_paths is None):
        raise ValueError("provide exactly one of image_path or image_paths")
    if image_paths is not None:
        if not image_paths or len({path.resolve() for path in image_paths}) != len(image_paths):
            raise ValueError("manifest image list must be nonempty and unique")
        if any(not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"} for path in image_paths):
            raise FileNotFoundError("manifest contains a missing or non-JPEG image")
        resolved_input: str | list[str] = [str(path.resolve()) for path in image_paths]
    else:
        assert image_path is not None
        if not image_path.is_dir():
            raise FileNotFoundError(image_path)
        resolved_input = str(image_path.resolve())
    return ReadOnlyJpegYOLODataset(
        img_path=resolved_input,
        imgsz=int(cfg.imgsz),
        batch_size=int(batch_size),
        augment=bool(augment),
        hyp=cfg,
        rect=bool(rect),
        cache=False,
        single_cls=False,
        stride=int(stride),
        pad=0.0 if augment else 0.5,
        prefix="Phase-B0: ",
        task="detect",
        classes=None,
        data=data,
        fraction=1.0,
    )
