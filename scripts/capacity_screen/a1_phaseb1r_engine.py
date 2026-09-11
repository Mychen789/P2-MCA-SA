from __future__ import annotations

"""Dormant manifest and single-cell execution adapter for Phase-B1R.

This module is import-safe and uses only the standard library.  It validates
the frozen manifest surface and builds an in-memory execution plan.  Numerical
imports and training remain disabled until formal AUTH/LOCK/PERMIT are issued.
"""

import csv
import hashlib
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


RUNTIME_ROOT = Path(__file__).resolve().parent
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

import a1_phaseb1r_contract as contract


ENGINE_ENABLED = False
ENGINE_BLOCK_REASON = "B1-R numerical execution remains disabled until formal authority is issued"

ATTACK_ROOT = RUNTIME_ROOT.parent
PAPER_ROOT = ATTACK_ROOT.parent
PROJECT_ROOT = PAPER_ROOT.parent
FREEZE_ROOT = ATTACK_ROOT / "A1_CAP_PhaseB0_data_freeze_20260821"
TRAIN_MANIFEST = FREEZE_ROOT / "phaseb1_train_manifest.tsv"
DEV_MANIFEST = FREEZE_ROOT / "phaseb1_dev_manifest.tsv"
DATA_ROOT = Path("D:/visdrone10_yolo").resolve()

MANIFEST_FIELDS = (
    "split",
    "ordinal",
    "sequence_id",
    "sequence_uid",
    "image_uid",
    "phaseb1_role",
    "image_rel",
    "label_rel",
    "image_bytes",
    "image_sha256",
    "label_bytes",
    "label_sha256",
    "width",
    "height",
    "raw_boxes",
    "effective_boxes",
    "duplicates_removed",
    "class_counts",
    "size_counts_768",
)


class B1REngineError(RuntimeError):
    """Raised when a frozen manifest or execution-plan boundary differs."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_relative(raw: str, *, prefix: tuple[str, str], label: str) -> PurePosixPath:
    value = PurePosixPath(raw)
    if value.is_absolute() or ".." in value.parts or value.parts[:2] != prefix:
        raise B1REngineError(f"unsafe {label}: {raw}")
    return value


def read_manifest_image_paths(
    manifest: str | Path,
    *,
    data_root: str | Path,
    expected_role: str,
    expected_rows: int,
    verify_files: bool = False,
) -> tuple[Path, ...]:
    """Validate one frozen TSV and return its unique image paths in row order."""
    path = Path(manifest).resolve()
    root = Path(data_root).resolve()
    if not path.is_file() or path.is_symlink():
        raise B1REngineError(f"manifest missing/non-regular: {path}")
    if expected_role not in {"TRAIN", "DEV"} or type(expected_rows) is not int or expected_rows <= 0:
        raise B1REngineError("manifest expectation is invalid")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise B1REngineError("manifest header is not exact")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise B1REngineError(f"manifest row count is {len(rows)}, expected {expected_rows}")

    images: list[Path] = []
    identities: set[str] = set()
    ordinals: set[int] = set()
    for index, row in enumerate(rows):
        if row.get("split") != "train" or row.get("phaseb1_role") != expected_role:
            raise B1REngineError(f"manifest role/split mismatch at row {index}")
        try:
            ordinal = int(row["ordinal"])
        except (KeyError, TypeError, ValueError) as exc:
            raise B1REngineError(f"invalid ordinal at row {index}") from exc
        if ordinal < 0 or ordinal in ordinals:
            raise B1REngineError(f"duplicate/negative ordinal at row {index}")
        ordinals.add(ordinal)
        image_rel = _safe_relative(row["image_rel"], prefix=("images", "train"), label="image_rel")
        label_rel = _safe_relative(row["label_rel"], prefix=("labels", "train"), label="label_rel")
        if image_rel.suffix.lower() not in {".jpg", ".jpeg"} or label_rel.suffix.lower() != ".txt":
            raise B1REngineError(f"manifest extension mismatch at row {index}")
        if image_rel.stem != label_rel.stem:
            raise B1REngineError(f"image/label stem mismatch at row {index}")
        image = (root / Path(*image_rel.parts)).resolve()
        label_path = (root / Path(*label_rel.parts)).resolve()
        for candidate, expected_parent, item_label in (
            (image, root / "images" / "train", "image"),
            (label_path, root / "labels" / "train", "label"),
        ):
            try:
                candidate.relative_to(expected_parent.resolve())
            except ValueError as exc:
                raise B1REngineError(f"{item_label} escapes the frozen data root") from exc
        identity = os.path.normcase(str(image))
        if identity in identities:
            raise B1REngineError(f"duplicate image identity at row {index}")
        identities.add(identity)
        if verify_files:
            if not image.is_file() or image.is_symlink() or not label_path.is_file() or label_path.is_symlink():
                raise B1REngineError(f"manifest source file missing/non-regular at row {index}")
            try:
                expected_image_bytes = int(row["image_bytes"])
                expected_label_bytes = int(row["label_bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise B1REngineError(f"manifest byte field invalid at row {index}") from exc
            if image.stat().st_size != expected_image_bytes or label_path.stat().st_size != expected_label_bytes:
                raise B1REngineError(f"manifest byte identity mismatch at row {index}")
            if file_sha256(image) != row.get("image_sha256") or file_sha256(label_path) != row.get("label_sha256"):
                raise B1REngineError(f"manifest SHA-256 identity mismatch at row {index}")
        images.append(image)
    return tuple(images)


def build_single_cell_plan(*, run_id: str, cell_id: str, result_root: str | Path) -> dict[str, Any]:
    """Build a no-write plan for exactly one frozen next cell."""
    if not isinstance(run_id, str) or not run_id:
        raise B1REngineError("run_id must be nonempty")
    if cell_id not in contract.CELL_ORDER:
        raise B1REngineError(f"unknown cell: {cell_id}")
    root = Path(result_root).resolve()
    return {
        "schema": "A1_PHASEB1R_SINGLE_CELL_PLAN_V1",
        "run_id": run_id,
        "cell_id": cell_id,
        "cell_index": contract.CELL_ORDER.index(cell_id),
        "cell_order": list(contract.CELL_ORDER),
        "train_manifest": str(TRAIN_MANIFEST.resolve()),
        "train_rows": 5149,
        "dev_access_authorized": False,
        "official_val_access_authorized": False,
        "checkpoint_path": str((root / contract.expected_checkpoint_name(cell_id)).resolve()),
        "training_commit_path": str((root / f"{cell_id}_TRAIN_COMMIT.json").resolve()),
        "automatic_follow_on": False,
        "numerical_execution_enabled": False,
    }


def main() -> None:
    raise B1REngineError(ENGINE_BLOCK_REASON)


if __name__ == "__main__":
    main()
