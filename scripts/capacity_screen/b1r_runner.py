from __future__ import annotations

"""Single simplified entry point for the local B1-R paper experiment.

One command is re-runnable.  It derives the next action from per-cell
``status.json`` files, supports interrupted-cell recovery, and evaluates only
after all four cells are complete.  Numerical execution remains disabled while
the simplified trainer/evaluator connection is being finalized.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


RUNTIME_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUNTIME_ROOT.parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "b1r_config.json"

RUN_ENABLED = True
RUN_BLOCK_REASON = "simplified B1-R runner is disabled"

CELL_ORDER = ("C10", "C11", "C01", "C00")
CELL_STATUS_SCHEMA = "B1R_CELL_STATUS_V1"
CELL_STATUS_KEYS = {
    "schema",
    "cell_id",
    "state",
    "completed_epochs",
    "recovery_checkpoint",
    "final_checkpoint",
    "checkpoint_strict_reload_passed",
    "dev_metric_computed",
}
CELL_STATES = {"RUNNING", "INTERRUPTED", "COMPLETE", "FAILED"}


class B1RRunnerError(RuntimeError):
    """Raised when the simplified experiment state is inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise B1RRunnerError(f"JSON missing/non-regular: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise B1RRunnerError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise B1RRunnerError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise B1RRunnerError(f"status temporary path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(dict(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cell_relative_path(cell_root: Path, raw: str | Path | None, *, label: str) -> str | None:
    if raw is None:
        return None
    candidate = Path(raw).resolve()
    try:
        relative = candidate.relative_to(cell_root.resolve())
    except ValueError as exc:
        raise B1RRunnerError(f"{label} escapes its cell directory") from exc
    return relative.as_posix()


def _status_document(
    *,
    cell_id: str,
    state: str,
    completed_epochs: int,
    recovery_checkpoint: str | None,
    final_checkpoint: str | None = None,
    strict_reload: bool = False,
) -> dict[str, Any]:
    document = {
        "schema": CELL_STATUS_SCHEMA,
        "cell_id": cell_id,
        "state": state,
        "completed_epochs": completed_epochs,
        "recovery_checkpoint": recovery_checkpoint,
        "final_checkpoint": final_checkpoint,
        "checkpoint_strict_reload_passed": strict_reload,
        "dev_metric_computed": False,
    }
    if set(document) != CELL_STATUS_KEYS:
        raise B1RRunnerError("internal status document keys changed")
    return document


def _project_path(raw: str, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise B1RRunnerError(f"{label} path is empty")
    candidate = (PROJECT_ROOT / raw).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise B1RRunnerError(f"{label} escapes the project root") from exc
    return candidate


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(Path(path).resolve())
    if set(config) != {"schema", "experiment", "cell_order", "train", "recovery", "final_checkpoint", "evaluation", "paths"}:
        raise B1RRunnerError("config top-level keys are not exact")
    if config["schema"] != "B1R_PAPER_EXPERIMENT_CONFIG_V1" or config["cell_order"] != list(CELL_ORDER):
        raise B1RRunnerError("config identity/cell order changed")
    train = config["train"]
    expected_train = {
        "seed": 42,
        "imgsz": 768,
        "batch": 2,
        "epochs": 150,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "cos_lr": True,
        "workers": 0,
        "amp": True,
        "early_stopping": False,
        "training_time_dev_validation": False,
    }
    if train != expected_train:
        raise B1RRunnerError("frozen training config changed")
    recovery = config["recovery"]
    if recovery != {
        "enabled": True,
        "save_every_epochs": 5,
        "resume_after_interruption": True,
        "recovery_checkpoint_used_for_model_selection": False,
    }:
        raise B1RRunnerError("recovery policy changed")
    if config["final_checkpoint"] != {
        "kind": "FINAL_EPOCH_EMA",
        "epoch_index": 149,
        "strict_reload_required": True,
    }:
        raise B1RRunnerError("final checkpoint policy changed")
    evaluation = config["evaluation"]
    if (
        evaluation.get("run_after_all_four_cells") is not True
        or evaluation.get("official_val_used_for_route_selection") is not False
        or evaluation.get("primary_metric") != "AP_S_768"
    ):
        raise B1RRunnerError("evaluation policy changed")
    paths = config["paths"]
    if set(paths) != {"train_manifest", "dev_manifest", "result_root"}:
        raise B1RRunnerError("config paths are not exact")
    return config


def configured_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    paths = config["paths"]
    return {
        "train_manifest": _project_path(paths["train_manifest"], label="train_manifest"),
        "dev_manifest": _project_path(paths["dev_manifest"], label="dev_manifest"),
        "result_root": _project_path(paths["result_root"], label="result_root"),
    }


def _optional_checkpoint(cell_root: Path, value: Any, *, label: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise B1RRunnerError(f"{label} must be null or a nonempty path")
    checkpoint = (cell_root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        checkpoint.relative_to(cell_root.resolve())
    except ValueError as exc:
        raise B1RRunnerError(f"{label} escapes its cell directory") from exc
    return checkpoint


def read_cell_status(cell_root: Path, cell_id: str) -> dict[str, Any] | None:
    if not cell_root.exists():
        return None
    if not cell_root.is_dir() or cell_root.is_symlink():
        raise B1RRunnerError(f"cell path is not a regular directory: {cell_root}")
    status = _read_json(cell_root / "status.json")
    if set(status) != CELL_STATUS_KEYS:
        raise B1RRunnerError(f"{cell_id} status keys are not exact")
    if status["schema"] != CELL_STATUS_SCHEMA or status["cell_id"] != cell_id or status["state"] not in CELL_STATES:
        raise B1RRunnerError(f"{cell_id} status identity/state is invalid")
    epochs = status["completed_epochs"]
    if type(epochs) is not int or not 0 <= epochs <= 150:
        raise B1RRunnerError(f"{cell_id} completed_epochs is invalid")
    recovery = _optional_checkpoint(cell_root, status["recovery_checkpoint"], label="recovery_checkpoint")
    final = _optional_checkpoint(cell_root, status["final_checkpoint"], label="final_checkpoint")
    if status["dev_metric_computed"] is not False:
        raise B1RRunnerError(f"{cell_id} contains a forbidden per-cell dev metric")
    if status["state"] == "COMPLETE":
        if epochs != 150 or final is None or not final.is_file() or final.is_symlink():
            raise B1RRunnerError(f"{cell_id} COMPLETE lacks its final checkpoint")
        if status["checkpoint_strict_reload_passed"] is not True:
            raise B1RRunnerError(f"{cell_id} final checkpoint strict reload did not pass")
    elif status["checkpoint_strict_reload_passed"] is not False or final is not None:
        raise B1RRunnerError(f"{cell_id} non-complete state claims a final checkpoint")
    if status["state"] in {"RUNNING", "INTERRUPTED"} and epochs > 0:
        if recovery is None or not recovery.is_file() or recovery.is_symlink():
            raise B1RRunnerError(f"{cell_id} recoverable state lacks a recovery checkpoint")
    return dict(status)


def inspect_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = configured_paths(config)
    root = paths["result_root"]
    completed: list[str] = []
    active: dict[str, Any] | None = None
    gap_seen = False
    for cell_id in CELL_ORDER:
        status = read_cell_status(root / cell_id, cell_id)
        if status is None:
            gap_seen = True
            continue
        if gap_seen:
            raise B1RRunnerError("cell directories are not a continuous frozen-order prefix")
        if status["state"] == "COMPLETE":
            completed.append(cell_id)
            continue
        active = status
        gap_seen = True

    if active is not None:
        action = "RESUME_CELL" if active["state"] in {"RUNNING", "INTERRUPTED"} else "INSPECT_OR_RETRY_CELL"
        return {
            "state": active["state"],
            "action": action,
            "cell_id": active["cell_id"],
            "completed_cells": completed,
            "dev_evaluation_allowed": False,
        }
    if len(completed) == len(CELL_ORDER):
        return {
            "state": "FOUR_CELLS_COMPLETE",
            "action": "EVALUATE_ALL_CELLS",
            "cell_id": None,
            "completed_cells": completed,
            "dev_evaluation_allowed": True,
        }
    next_cell = CELL_ORDER[len(completed)]
    return {
        "state": "READY_FOR_NEXT_CELL",
        "action": "TRAIN_NEXT_CELL",
        "cell_id": next_cell,
        "completed_cells": completed,
        "dev_evaluation_allowed": False,
    }


def run_once(config: Mapping[str, Any], *, executor: Any = None) -> dict[str, Any]:
    """Run or resume one cell, or evaluate after four cells, once enabled."""
    state = inspect_experiment(config)
    if RUN_ENABLED is not True:
        raise B1RRunnerError(RUN_BLOCK_REASON)
    if state["action"] == "EVALUATE_ALL_CELLS":
        raise B1RRunnerError("four-cell evaluation adapter is not connected yet")
    if state["action"] not in {"TRAIN_NEXT_CELL", "RESUME_CELL", "INSPECT_OR_RETRY_CELL"}:
        raise B1RRunnerError(f"unsupported action: {state['action']}")

    paths = configured_paths(config)
    result_root = paths["result_root"]
    cell_id = state["cell_id"]
    cell_root = (result_root / cell_id).resolve()
    status_path = cell_root / "status.json"
    previous = read_cell_status(cell_root, cell_id)
    completed_epochs = 0 if previous is None else previous["completed_epochs"]
    recovery_value = None if previous is None else previous["recovery_checkpoint"]
    recovery_path = _optional_checkpoint(cell_root, recovery_value, label="recovery_checkpoint")
    _write_json_atomic(
        status_path,
        _status_document(
            cell_id=cell_id,
            state="RUNNING",
            completed_epochs=completed_epochs,
            recovery_checkpoint=recovery_value,
        ),
    )

    if executor is None:
        if str(RUNTIME_ROOT) not in sys.path:
            sys.path.insert(0, str(RUNTIME_ROOT))
        import a1_phaseb1r_numeric

        executor = a1_phaseb1r_numeric.execute_cell
    if not callable(executor):
        raise B1RRunnerError("cell executor is not callable")

    latest = {
        "completed_epochs": completed_epochs,
        "recovery_checkpoint": recovery_value,
        "state": "RUNNING",
    }

    def progress(update: Mapping[str, Any]) -> None:
        epochs = update.get("completed_epochs")
        if type(epochs) is not int or not 1 <= epochs <= 150:
            raise B1RRunnerError("executor reported an invalid completed epoch count")
        recovery = _cell_relative_path(
            cell_root,
            update.get("recovery_checkpoint"),
            label="recovery_checkpoint",
        )
        final_raw = update.get("final_checkpoint")
        strict = update.get("checkpoint_strict_reload_passed") is True
        complete = epochs == 150 and final_raw is not None and strict
        final = _cell_relative_path(cell_root, final_raw, label="final_checkpoint") if complete else None
        latest.update(
            {
                "completed_epochs": epochs,
                "recovery_checkpoint": recovery,
                "state": "COMPLETE" if complete else "RUNNING",
            }
        )
        _write_json_atomic(
            status_path,
            _status_document(
                cell_id=cell_id,
                state=latest["state"],
                completed_epochs=epochs,
                recovery_checkpoint=recovery,
                final_checkpoint=final,
                strict_reload=complete,
            ),
        )

    try:
        result = executor(
            run_id=str(config["experiment"]),
            cell_id=cell_id,
            result_root=result_root,
            recovery_checkpoint=recovery_path,
            verify_manifest_files=previous is None,
            progress_callback=progress,
        )
        if not isinstance(result, Mapping):
            raise B1RRunnerError("cell executor returned a non-mapping result")
        checkpoint_qc = result.get("checkpoint_qc")
        if not isinstance(checkpoint_qc, Mapping) or checkpoint_qc.get("passed") is not True:
            raise B1RRunnerError("cell executor returned without passing final checkpoint QC")
        final = _cell_relative_path(cell_root, checkpoint_qc.get("path"), label="final_checkpoint")
        recovery = _cell_relative_path(
            cell_root,
            result.get("recovery_checkpoint"),
            label="recovery_checkpoint",
        )
        final_document = _status_document(
            cell_id=cell_id,
            state="COMPLETE",
            completed_epochs=150,
            recovery_checkpoint=recovery,
            final_checkpoint=final,
            strict_reload=True,
        )
        _write_json_atomic(status_path, final_document)
        return {
            "state": "CELL_COMPLETE",
            "action": "STOP_AFTER_ONE_CELL",
            "cell_id": cell_id,
            "completed_epochs": 150,
            "resumed_from_epoch": result.get("resumed_from_epoch"),
            "amp_skips": result.get("amp_skips", 0),
        }
    except BaseException:
        current = read_cell_status(cell_root, cell_id)
        if current is None or current["state"] != "COMPLETE":
            last_path = cell_root / "weights" / "last.pt"
            recovery = latest["recovery_checkpoint"]
            if last_path.is_file() and not last_path.is_symlink():
                recovery = _cell_relative_path(cell_root, last_path, label="recovery_checkpoint")
            _write_json_atomic(
                status_path,
                _status_document(
                    cell_id=cell_id,
                    state="INTERRUPTED",
                    completed_epochs=latest["completed_epochs"],
                    recovery_checkpoint=recovery,
                ),
            )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified re-runnable B1-R paper experiment")
    parser.add_argument("--status", action="store_true", help="inspect state without training")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config()
    state = inspect_experiment(config)
    if args.status:
        print(json.dumps(state, ensure_ascii=False, sort_keys=True))
        return 0
    if RUN_ENABLED is not True:
        print(RUN_BLOCK_REASON, file=sys.stderr)
        print(json.dumps(state, ensure_ascii=False, sort_keys=True))
        return 2
    run_once(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
