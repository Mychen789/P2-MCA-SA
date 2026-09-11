from __future__ import annotations

"""Pure contracts for the future Phase-B1R four-cell transaction.

The module has no launcher and imports no numerical library.  It defines the
formal cell order, exact training-COMMIT contract, and the fail-closed rule that
keeps dev evaluation locked until all four blinded training COMMITs validate.
"""

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


CELL_ORDER = ("C10", "C11", "C01", "C00")
CELL_SPECS: dict[str, dict[str, Any]] = {
    "C00": {"p2": False, "c3": 96, "params_total": 24_116_254, "strides": [8, 16, 32]},
    "C01": {"p2": False, "c3": 192, "params_total": 25_862_110, "strides": [8, 16, 32]},
    "C10": {"p2": True, "c3": 96, "params_total": 25_055_336, "strides": [4, 8, 16, 32]},
    "C11": {"p2": True, "c3": 192, "params_total": 27_134_312, "strides": [4, 8, 16, 32]},
}
INITIAL_STATE_SHA256: dict[str, str] = {
    "C00": "F093CE268CC9DE5A4A337D2799E69866D63F71A3E80973C845759912F42C870A",
    "C01": "25260F86221123AF2FD642919C6AAFAD232F06D1ACB1AEEE105EA3A701DFC311",
    "C10": "9C4AD18BE6DC82FE321403C27C1281BB713A01530726FA9778A4C7CFDD524191",
    "C11": "5FD55FD759811B0D4EA944411F92D09B5C848A370BC4462AEA1881CA902B2BAD",
}

TRAIN_COMMIT_SCHEMA = "A1_PHASEB1R_CELL_TRAIN_COMMIT_V1"
TRAIN_COMMIT_STATUS = "B1R_CELL_TRAINING_COMMITTED_BLINDED"
EVALUATION_UNLOCK_SCHEMA = "A1_PHASEB1R_EVALUATION_UNLOCK_V1"
EVALUATION_UNLOCK_STATUS = "AUTHORIZED_BLINDED_DEV_EVALUATION_AFTER_FOUR_TRAINING_COMMITS"

FROZEN_TRAINING = {
    "seed": 42,
    "imgsz": 768,
    "batch": 2,
    "epochs_completed": 150,
    "last_epoch_index": 149,
    "optimizer_family": "SGD",
    "amp": True,
    "workers": 0,
}

TRAIN_COMMIT_KEYS = {
    "schema",
    "status",
    "run_id",
    "cell_id",
    "cell_index",
    "cell_order",
    "seed",
    "imgsz",
    "batch",
    "epochs_completed",
    "last_epoch_index",
    "optimizer_family",
    "amp",
    "workers",
    "early_stopping_used",
    "training_time_dev_accessed",
    "official_val_accessed",
    "performance_metric_computed",
    "source_dataset_write",
    "initial_state_sha256",
    "checkpoint",
    "commit_body_sha256",
}

CHECKPOINT_KEYS = {
    "name",
    "path",
    "bytes",
    "sha256",
    "kind",
    "epoch_index",
    "ema",
    "strict_reload_passed",
    "state_sha256",
}

FORBIDDEN_TRAIN_RESULT_KEYS = {
    "ap",
    "ap50",
    "ap75",
    "ap_small",
    "apsmall",
    "map",
    "map50",
    "map75",
    "precision",
    "recall",
    "fitness",
    "metrics",
    "effect_direction",
    "gate_result",
}


class B1RContractError(RuntimeError):
    """Raised when a future B1-R transaction violates the frozen design."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest().upper()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def with_body_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    if field in result:
        raise B1RContractError(f"body already contains {field}")
    result[field] = canonical_sha256(result)
    return result


def validate_body_hash(value: Mapping[str, Any], field: str) -> None:
    digest = value.get(field)
    if not _is_sha256(digest):
        raise B1RContractError(f"missing/invalid {field}")
    body = dict(value)
    body.pop(field)
    if canonical_sha256(body) != digest:
        raise B1RContractError(f"{field} mismatch")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789ABCDEF" for character in value)
    )


def _require_exact_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise B1RContractError(f"{label} keys are not exact")
    return value


def _reject_nonfinite(value: Any, label: str = "document") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise B1RContractError(f"non-finite number in {label}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{label}[{index}]")


def _reject_training_result_leak(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_TRAIN_RESULT_KEYS:
                raise B1RContractError(f"training COMMIT leaks result/effect field: {key}")
            _reject_training_result_leak(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_training_result_leak(child)


def derive_next_cell(completed_cells: Sequence[str]) -> str | None:
    observed = tuple(completed_cells)
    if observed != CELL_ORDER[: len(observed)] or len(observed) > len(CELL_ORDER):
        raise B1RContractError("completed cells are not an exact frozen-order prefix")
    return CELL_ORDER[len(observed)] if len(observed) < len(CELL_ORDER) else None


def expected_checkpoint_name(cell_id: str) -> str:
    if cell_id not in CELL_SPECS:
        raise B1RContractError(f"unknown B1-R cell: {cell_id}")
    return f"{cell_id}_FINAL_EPOCH_EMA_EPOCH_149.pt"


def build_training_commit(
    *,
    run_id: str,
    cell_id: str,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(run_id, str) or not run_id:
        raise B1RContractError("run_id must be a nonempty string")
    if cell_id not in CELL_SPECS:
        raise B1RContractError(f"unknown B1-R cell: {cell_id}")
    body = {
        "schema": TRAIN_COMMIT_SCHEMA,
        "status": TRAIN_COMMIT_STATUS,
        "run_id": run_id,
        "cell_id": cell_id,
        "cell_index": CELL_ORDER.index(cell_id),
        "cell_order": list(CELL_ORDER),
        **FROZEN_TRAINING,
        "early_stopping_used": False,
        "training_time_dev_accessed": False,
        "official_val_accessed": False,
        "performance_metric_computed": False,
        "source_dataset_write": False,
        "initial_state_sha256": INITIAL_STATE_SHA256[cell_id],
        "checkpoint": dict(checkpoint),
    }
    return with_body_hash(body, "commit_body_sha256")


def validate_training_commit(
    commit: Mapping[str, Any],
    *,
    expected_cell: str,
    expected_run_id: str | None = None,
    verify_checkpoint_file: bool = False,
    expected_cell_root: str | Path | None = None,
) -> None:
    _require_exact_keys(commit, TRAIN_COMMIT_KEYS, "training COMMIT")
    validate_body_hash(commit, "commit_body_sha256")
    _reject_nonfinite(commit)
    body_for_leak_check = {
        key: value
        for key, value in commit.items()
        if key != "performance_metric_computed"
    }
    _reject_training_result_leak(body_for_leak_check)
    if expected_cell not in CELL_SPECS or commit.get("cell_id") != expected_cell:
        raise B1RContractError("training COMMIT cell mismatch")
    if expected_run_id is not None and commit.get("run_id") != expected_run_id:
        raise B1RContractError("training COMMIT run_id mismatch")
    if (
        commit.get("schema") != TRAIN_COMMIT_SCHEMA
        or commit.get("status") != TRAIN_COMMIT_STATUS
        or commit.get("cell_index") != CELL_ORDER.index(expected_cell)
        or commit.get("cell_order") != list(CELL_ORDER)
        or commit.get("initial_state_sha256") != INITIAL_STATE_SHA256[expected_cell]
    ):
        raise B1RContractError("training COMMIT identity/order mismatch")
    for key, expected in FROZEN_TRAINING.items():
        if commit.get(key) != expected or type(commit.get(key)) is not type(expected):
            raise B1RContractError(f"training COMMIT frozen field mismatch: {key}")
    for key in (
        "early_stopping_used",
        "training_time_dev_accessed",
        "official_val_accessed",
        "performance_metric_computed",
        "source_dataset_write",
    ):
        if commit.get(key) is not False:
            raise B1RContractError(f"training-only boundary violated: {key}")

    checkpoint = _require_exact_keys(commit.get("checkpoint"), CHECKPOINT_KEYS, "checkpoint")
    if (
        checkpoint.get("name") != expected_checkpoint_name(expected_cell)
        or checkpoint.get("kind") != "FINAL_EPOCH_EMA"
        or checkpoint.get("epoch_index") != 149
        or checkpoint.get("ema") is not True
        or checkpoint.get("strict_reload_passed") is not True
        or type(checkpoint.get("bytes")) is not int
        or checkpoint["bytes"] <= 0
        or not _is_sha256(checkpoint.get("sha256"))
        or not _is_sha256(checkpoint.get("state_sha256"))
    ):
        raise B1RContractError("final-epoch EMA checkpoint contract mismatch")
    path = Path(str(checkpoint.get("path", ""))).resolve()
    if path.name != checkpoint["name"]:
        raise B1RContractError("checkpoint path/name mismatch")
    if expected_cell_root is not None:
        root = Path(expected_cell_root).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise B1RContractError("checkpoint escapes the cell transaction root") from exc
    if verify_checkpoint_file:
        if not path.is_file() or path.is_symlink():
            raise B1RContractError("checkpoint file missing/non-regular")
        if path.stat().st_size != checkpoint["bytes"] or file_sha256(path) != checkpoint["sha256"]:
            raise B1RContractError("checkpoint file identity mismatch")


def validate_training_prefix(
    commits: Sequence[Mapping[str, Any]],
    *,
    expected_run_id: str,
) -> tuple[str, ...]:
    if len(commits) > len(CELL_ORDER):
        raise B1RContractError("more than four B1-R training COMMITs")
    completed: list[str] = []
    for index, commit in enumerate(commits):
        cell = CELL_ORDER[index]
        validate_training_commit(commit, expected_cell=cell, expected_run_id=expected_run_id)
        completed.append(cell)
    derive_next_cell(completed)
    return tuple(completed)


def build_evaluation_unlock(
    commit_references: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
) -> dict[str, Any]:
    if not isinstance(run_id, str) or not run_id:
        raise B1RContractError("run_id must be a nonempty string")
    if len(commit_references) != len(CELL_ORDER):
        raise B1RContractError("dev evaluation requires exactly four training COMMIT references")
    normalized: list[dict[str, Any]] = []
    for cell, reference in zip(CELL_ORDER, commit_references):
        row = _require_exact_keys(reference, {"cell_id", "path", "bytes", "sha256", "body_sha256"}, "COMMIT reference")
        if (
            row.get("cell_id") != cell
            or type(row.get("bytes")) is not int
            or row["bytes"] <= 0
            or not _is_sha256(row.get("sha256"))
            or not _is_sha256(row.get("body_sha256"))
        ):
            raise B1RContractError("training COMMIT reference identity mismatch")
        normalized.append(dict(row))
    body = {
        "schema": EVALUATION_UNLOCK_SCHEMA,
        "status": EVALUATION_UNLOCK_STATUS,
        "run_id": run_id,
        "cell_order": list(CELL_ORDER),
        "training_commit_references": normalized,
        "training_commit_closure_sha256": canonical_sha256(normalized),
        "dev_access_authorized": True,
        "official_val_access_authorized": False,
        "checkpoint_selection_authorized": False,
        "effect_direction_release_authorized": False,
        "evaluation_scope": "ONE_BLINDED_DEV_PASS_ALL_FOUR_FINAL_EMA_CHECKPOINTS",
    }
    return with_body_hash(body, "unlock_body_sha256")


def next_action(commits: Sequence[Mapping[str, Any]], *, run_id: str) -> dict[str, Any]:
    completed = validate_training_prefix(commits, expected_run_id=run_id)
    cell = derive_next_cell(completed)
    if cell is not None:
        return {
            "action": "TRAIN_NEXT_CELL",
            "cell_id": cell,
            "cell_index": CELL_ORDER.index(cell),
            "dev_access_authorized": False,
            "performance_metric_authorized": False,
        }
    return {
        "action": "BUILD_EVALUATION_UNLOCK",
        "cell_id": None,
        "cell_index": None,
        "dev_access_authorized": False,
        "reason": "FOUR_VALID_TRAINING_COMMITS_REQUIRE_SEPARATE_UNLOCK_TRANSACTION",
    }


__all__ = [
    "B1RContractError",
    "CELL_ORDER",
    "CELL_SPECS",
    "EVALUATION_UNLOCK_SCHEMA",
    "FROZEN_TRAINING",
    "INITIAL_STATE_SHA256",
    "TRAIN_COMMIT_SCHEMA",
    "build_evaluation_unlock",
    "build_training_commit",
    "canonical_sha256",
    "derive_next_cell",
    "expected_checkpoint_name",
    "next_action",
    "validate_training_commit",
    "validate_training_prefix",
    "with_body_hash",
]

