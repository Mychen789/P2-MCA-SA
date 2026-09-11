from __future__ import annotations

"""Numerical wiring for one cell of the simplified Phase-B1R runner.

The module is third-party-import free at import time.  The outer runner owns
the only execution switch.  This adapter connects the frozen B0 builder,
manifest dataset, recoverable Ultralytics trainer, and final checkpoint QC.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping


RUNTIME_ROOT = Path(__file__).resolve().parent
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

import a1_phaseb1r_cell as cell_policy
import a1_phaseb1r_contract as contract
import a1_phaseb1r_engine as engine


PROJECT_ROOT = RUNTIME_ROOT.parent.parent.parent
BUILDER_PATH = RUNTIME_ROOT / "a1_phaseb0_builder.py"
BUILDER_BYTES = 25_888
BUILDER_SHA256 = "549E39D4E90785300FA1B3E154F8E8840305DC468FC9E95E9CDB29E60D9C7C9F"
DATA_ADAPTER_PATH = RUNTIME_ROOT / "a1_phaseb0_data.py"
DATA_ADAPTER_BYTES = 7_942
DATA_ADAPTER_SHA256 = "276736B770B9A8F089C56F8B79846E5736F38F85591B67FB31B4368989E78619"
P2_YAML = PROJECT_ROOT / "configs" / "models" / "yolov8m-p2.yaml"
MODEL_YAML = Path(
    "C:/Users/cmy/anaconda3/envs/yolov8/Lib/site-packages/ultralytics/cfg/models/v8/yolov8.yaml"
)
DATA_YAML = PROJECT_ROOT / "configs" / "data" / "visdrone10.yaml"
WEIGHTS = PROJECT_ROOT / "weights" / "yolov8m.pt"
class B1RNumericError(RuntimeError):
    """Raised when the numerical stack or frozen cell identity differs."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_local_source(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise B1RNumericError(f"local source missing/non-regular: {path}")
    if path.stat().st_size != expected_bytes or file_sha256(path) != expected_sha256:
        raise B1RNumericError(f"local source identity mismatch: {path}")


def _load_local_module(name: str, path: Path) -> Any:
    if name in sys.modules:
        raise B1RNumericError(f"local module name already occupied: {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise B1RNumericError(f"cannot load local module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def lazy_import_numeric_stack() -> dict[str, Any]:
    """Import the exact numeric surface only after an authority gate calls it."""
    require_local_source(BUILDER_PATH, expected_bytes=BUILDER_BYTES, expected_sha256=BUILDER_SHA256)
    require_local_source(
        DATA_ADAPTER_PATH,
        expected_bytes=DATA_ADAPTER_BYTES,
        expected_sha256=DATA_ADAPTER_SHA256,
    )
    import numpy as np
    import torch
    from ultralytics.data import build_dataloader
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils import YAML
    from ultralytics.utils.callbacks import get_default_callbacks

    return {
        "np": np,
        "torch": torch,
        "build_dataloader": build_dataloader,
        "DetectionTrainer": DetectionTrainer,
        "YAML": YAML,
        "get_default_callbacks": get_default_callbacks,
        "builder": _load_local_module("a1_phaseb0_builder_b1r_numeric", BUILDER_PATH),
        "data_adapter": _load_local_module("a1_phaseb0_data_b1r_numeric", DATA_ADAPTER_PATH),
    }


def _data_document(stack: Mapping[str, Any]) -> dict[str, Any]:
    document = stack["YAML"].load(DATA_YAML)
    names = document.get("names")
    if isinstance(names, list):
        names = {index: value for index, value in enumerate(names)}
    if not isinstance(names, Mapping) or len(names) != 10:
        raise B1RNumericError("VisDrone data YAML does not define exactly ten names")
    return {"names": dict(names), "nc": 10, "channels": 3}


def build_fresh_cell(stack: Mapping[str, Any], cell_id: str) -> tuple[Any, dict[str, Any]]:
    """Build and attest one fresh B0-equivalent cell; no training occurs here."""
    required = {"torch", "builder"}
    if not required.issubset(stack) or cell_id not in contract.CELL_ORDER:
        raise B1RNumericError("numeric stack/cell identity is invalid")
    builder = stack["builder"]
    cells = builder.build_cells(P2_YAML, training_seed=42, official_yaml=MODEL_YAML)
    if set(cells) != set(contract.CELL_ORDER):
        raise B1RNumericError("B0 builder did not return exactly four frozen cells")
    transfer = builder.transfer_common_pretrained(cells, WEIGHTS)
    invariance = builder.verify_pair_invariance(cells)
    selected = cells[cell_id]
    model = selected.model.cpu()
    state_sha256 = cell_policy.model_state_sha256(stack["torch"], model)
    params_total = sum(int(parameter.numel()) for parameter in model.parameters())
    strides = [int(float(value)) for value in model.model[-1].stride]
    cell_policy.validate_cell_identity(
        cell_id,
        state_sha256=state_sha256,
        params_total=params_total,
        strides=strides,
    )
    for other_id, built in list(cells.items()):
        if other_id != cell_id:
            built.model.cpu()
            del cells[other_id]
    return model, {
        "schema": "A1_PHASEB1R_FRESH_CELL_QC_V1",
        "cell_id": cell_id,
        "initial_state_sha256": state_sha256,
        "params_total": params_total,
        "strides": strides,
        "transfer_schema": transfer.get("schema"),
        "invariance_schema": invariance.get("semantic_body", {}).get("schema"),
    }


def prepare_trainer(
    stack: Mapping[str, Any],
    *,
    run_id: str,
    cell_id: str,
    result_root: str | Path,
    recovery_checkpoint: str | Path | None,
    verify_manifest_files: bool,
    progress_callback: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Prepare one fresh or recoverable trainer for the selected cell."""
    expected_stack = {
        "np",
        "torch",
        "build_dataloader",
        "DetectionTrainer",
        "YAML",
        "get_default_callbacks",
        "builder",
        "data_adapter",
    }
    if set(stack) != expected_stack:
        raise B1RNumericError("numeric stack keys are not exact")
    root = Path(result_root).resolve()
    cell_root = (root / cell_id).resolve()
    try:
        cell_root.relative_to(root)
    except ValueError as exc:
        raise B1RNumericError("cell path escapes result root") from exc
    plan = engine.build_single_cell_plan(run_id=run_id, cell_id=cell_id, result_root=cell_root)
    image_paths = engine.read_manifest_image_paths(
        engine.TRAIN_MANIFEST,
        data_root=engine.DATA_ROOT,
        expected_role="TRAIN",
        expected_rows=5149,
        verify_files=verify_manifest_files,
    )
    model, fresh_qc = build_fresh_cell(stack, cell_id)

    def fresh_model_factory() -> Any:
        fresh, _report = build_fresh_cell(stack, cell_id)
        return fresh

    trainer_class = cell_policy.build_trainer_class(
        {
            "DetectionTrainer": stack["DetectionTrainer"],
            "build_dataloader": stack["build_dataloader"],
            "data_adapter": stack["data_adapter"],
            "torch": stack["torch"],
            "np": stack["np"],
        }
    )
    dynamic = {
        "model": str(MODEL_YAML.resolve()),
        "data": str(DATA_YAML.resolve()),
        "project": str(root),
        "name": cell_id,
    }
    callbacks = stack["get_default_callbacks"]()
    if progress_callback is not None:
        if not callable(progress_callback):
            raise B1RNumericError("progress_callback must be callable")

        def publish_progress(current_trainer):
            update = {
                "completed_epochs": int(current_trainer.epoch) + 1,
                "recovery_checkpoint": str(Path(current_trainer.last).resolve()),
                "amp_skips": int(current_trainer.phaseb1r_amp_skips),
            }
            qc = getattr(current_trainer, "phaseb1r_checkpoint_qc", None)
            if update["completed_epochs"] == 150 and isinstance(qc, Mapping) and qc.get("passed") is True:
                update["final_checkpoint"] = qc["path"]
                update["checkpoint_strict_reload_passed"] = True
            progress_callback(update)

        callbacks["on_model_save"].append(publish_progress)
    results_path = cell_root / "results.csv"
    results_backup = None
    if results_path.is_symlink():
        raise B1RNumericError("results.csv must not be a symlink")
    if results_path.is_file():
        results_backup = results_path.read_bytes()
    try:
        trainer = trainer_class(
            context={
                "cell_id": cell_id,
                "model": model,
                "image_paths": image_paths,
                "data": _data_document(stack),
                "cell_root": str(cell_root),
                "recovery_checkpoint": None if recovery_checkpoint is None else str(Path(recovery_checkpoint).resolve()),
                "checkpoint_path": plan["checkpoint_path"],
                "fresh_model_factory": fresh_model_factory,
            },
            overrides=cell_policy.expected_overrides(dynamic),
            callbacks=callbacks,
        )
    finally:
        if results_backup is not None and not results_path.exists():
            results_path.write_bytes(results_backup)
    return trainer, {**plan, "fresh_cell_qc": fresh_qc}


def execute_cell(
    *,
    run_id: str,
    cell_id: str,
    result_root: str | Path,
    recovery_checkpoint: str | Path | None,
    verify_manifest_files: bool,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Train/resume one cell.  The outer runner must gate all calls here."""
    stack = lazy_import_numeric_stack()
    trainer, plan = prepare_trainer(
        stack,
        run_id=run_id,
        cell_id=cell_id,
        result_root=result_root,
        recovery_checkpoint=recovery_checkpoint,
        verify_manifest_files=verify_manifest_files,
        progress_callback=progress_callback,
    )
    recovery_epoch = None
    if recovery_checkpoint is not None:
        recovery_payload = stack["torch"].load(
            Path(recovery_checkpoint).resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(recovery_payload, Mapping):
            recovery_epoch = recovery_payload.get("epoch")
    if recovery_epoch == 149:
        trainer._setup_train()
        trainer.publish_final_checkpoint()
        if progress_callback is not None:
            progress_callback(
                {
                    "completed_epochs": 150,
                    "recovery_checkpoint": str(Path(trainer.last).resolve()),
                    "amp_skips": int(trainer.phaseb1r_amp_skips),
                    "final_checkpoint": trainer.phaseb1r_checkpoint_qc["path"],
                    "checkpoint_strict_reload_passed": True,
                }
            )
    else:
        trainer.train()
    qc = getattr(trainer, "phaseb1r_checkpoint_qc", None)
    if not isinstance(qc, Mapping) or qc.get("passed") is not True:
        raise B1RNumericError("training returned without a passing checkpoint QC")
    return {
        **plan,
        "checkpoint_qc": dict(qc),
        "recovery_checkpoint": str(Path(trainer.last).resolve()),
        "resumed_from_epoch": trainer.phaseb1r_resumed_from_epoch,
        "amp_skips": int(trainer.phaseb1r_amp_skips),
    }


def main() -> None:
    raise B1RNumericError("call this adapter only through b1r_runner.py")


if __name__ == "__main__":
    main()
