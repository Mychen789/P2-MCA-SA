from __future__ import annotations

"""Fail-closed policy core for the future Phase-B1R cell child.

This file intentionally stays import-safe and dormant.  It freezes the exact
Ultralytics overrides and the already validated DFL optimizer correction while
the numerical trainer factory and parent authority envelope are completed.
"""

import hashlib
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


RUNTIME_ROOT = Path(__file__).resolve().parent
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

import a1_phaseb1r_contract as contract


CELL_ENGINE_IMPLEMENTED = False
CELL_ENGINE_BLOCK_REASON = (
    "B1-R cell numerical execution is disabled until trainer, envelope, checkpoint transaction "
    "and fault tests are complete"
)

# The Detect module's index depends on the P2 switch, so the layer number is a
# property of the cell and must not be part of the lock.  P2=on inserts six head
# layers and puts Detect at model.28 (C10/C11); P2=off leaves it at model.22
# (C01/C00).  The B0 closure records both names.  Locking the full name to the
# P2=on value made C01 and C00 abort inside build_optimizer before the first
# batch (2026-09-09).  What this guard exists to assert is that the single frozen
# parameter is the DFL convolution -- established by the suffix, shape, element
# count and requires_grad flag, none of which depend on the layer index.
LOCKED_DFL_SUFFIX = ".dfl.conv.weight"
LOCKED_DFL_NAME = "model.28.dfl.conv.weight"  # P2=on reference name; retained for the contract tests
LOCKED_DFL_SHAPE = (1, 16, 1, 1)
LOCKED_DFL_NUMEL = 16
LOCKED_OPTIMIZER_GROUPS = 3

# Values omitted from the old documents keep the locked Ultralytics 8.4.37
# default.  In particular nbs=64 preserves the standard warmup accumulation
# schedule instead of inventing a new effective batch policy.
TRAINING_OVERRIDES: dict[str, Any] = {
    "task": "detect",
    "device": 0,
    "seed": 42,
    "deterministic": True,
    "imgsz": 768,
    "batch": 2,
    "epochs": 150,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "cos_lr": True,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "nbs": 64,
    "amp": True,
    "workers": 0,
    "cache": False,
    "rect": False,
    "close_mosaic": 15,
    "mosaic": 1.0,
    "mixup": 0.15,
    "cutmix": 0.0,
    "copy_paste": 0.3,
    "copy_paste_mode": "flip",
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "bgr": 0.0,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "patience": 0,
    "val": False,
    "save": True,
    "save_period": 5,
    "plots": False,
    "resume": False,
    "exist_ok": True,
    "pretrained": False,
    "freeze": None,
    "compile": False,
    "multi_scale": 0.0,
    "fraction": 1.0,
    "single_cls": False,
    "classes": None,
    "augmentations": [],
}

DYNAMIC_OVERRIDE_KEYS = {"model", "data", "project", "name"}


def tensor_sha256(torch_module: Any, tensor: Any) -> str:
    """Hash tensor metadata and contiguous CPU bytes for checkpoint QC."""
    value = tensor.detach().cpu().contiguous()
    header = contract.canonical_bytes(
        {"dtype": str(value.dtype), "shape": list(value.shape)}
    ) + b"\0"
    raw = value.reshape(-1).view(torch_module.uint8).numpy().tobytes()
    return hashlib.sha256(header + raw).hexdigest().upper()


def state_mapping_sha256(torch_module: Any, state: Mapping[str, Any]) -> str:
    rows = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_sha256(torch_module, value),
        }
        for name, value in state.items()
    ]
    return contract.canonical_sha256(rows)


def model_state_sha256(torch_module: Any, model: Any) -> str:
    return state_mapping_sha256(torch_module, model.state_dict())


class B1RCellError(RuntimeError):
    """Raised when a cell-child policy differs from the frozen screen."""


def expected_overrides(dynamic: Mapping[str, Any]) -> dict[str, Any]:
    if set(dynamic) != DYNAMIC_OVERRIDE_KEYS:
        raise B1RCellError("dynamic override keys are not exact")
    values = dict(dynamic)
    if not all(isinstance(values[key], (str, Path)) and str(values[key]) for key in DYNAMIC_OVERRIDE_KEYS):
        raise B1RCellError("dynamic override values must be nonempty paths/names")
    return {**TRAINING_OVERRIDES, **{key: str(values[key]) for key in DYNAMIC_OVERRIDE_KEYS}}


def validate_overrides(observed: Mapping[str, Any], dynamic: Mapping[str, Any]) -> None:
    expected = expected_overrides(dynamic)
    if dict(observed) != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(key for key in set(expected) & set(observed) if observed[key] != expected[key])
        raise B1RCellError(f"training overrides drifted: missing={missing}, extra={extra}, changed={changed}")


def validate_cell_identity(cell_id: str, *, state_sha256: str, params_total: int, strides: list[int]) -> None:
    if cell_id not in contract.CELL_ORDER:
        raise B1RCellError(f"unknown cell: {cell_id}")
    spec = contract.CELL_SPECS[cell_id]
    if (
        state_sha256 != contract.INITIAL_STATE_SHA256[cell_id]
        or type(params_total) is not int
        or params_total != spec["params_total"]
        or strides != spec["strides"]
    ):
        raise B1RCellError(f"fresh {cell_id} identity differs from the B0 closure")


def _optimizer_parameters(optimizer: Any) -> list[Any]:
    try:
        groups = list(optimizer.param_groups)
    except (AttributeError, TypeError) as exc:
        raise B1RCellError("optimizer param_groups surface is absent") from exc
    if len(groups) != LOCKED_OPTIMIZER_GROUPS:
        raise B1RCellError("optimizer group count is not the locked Ultralytics three-group layout")
    parameters: list[Any] = []
    for group in groups:
        if not isinstance(group, dict) or "params" not in group:
            raise B1RCellError("optimizer group is malformed")
        parameters.extend(list(group["params"]))
    return parameters


def exclude_locked_dfl_after_standard_build(model: Any, optimizer: Any) -> dict[str, Any]:
    try:
        named = list(model.named_parameters())
    except (AttributeError, TypeError) as exc:
        raise B1RCellError("model named_parameters surface is absent") from exc
    frozen = [(name, parameter) for name, parameter in named if getattr(parameter, "requires_grad", None) is False]
    if len(frozen) != 1:
        raise B1RCellError(f"expected exactly one frozen parameter, observed {len(frozen)}")
    name, parameter = frozen[0]
    try:
        shape = tuple(int(value) for value in parameter.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise B1RCellError("frozen parameter shape is unreadable") from exc
    if (
        not name.endswith(LOCKED_DFL_SUFFIX)
        or shape != LOCKED_DFL_SHAPE
        or math.prod(shape) != LOCKED_DFL_NUMEL
        or getattr(parameter, "requires_grad", None) is not False
    ):
        raise B1RCellError(
            "frozen parameter is not the locked DFL convolution: "
            f"name={name!r} shape={shape!r} requires_grad={getattr(parameter, 'requires_grad', None)!r} "
            f"(expected a *{LOCKED_DFL_SUFFIX} parameter of shape {LOCKED_DFL_SHAPE} with requires_grad=False)"
        )

    before = _optimizer_parameters(optimizer)
    occurrences = sum(candidate is parameter for candidate in before)
    if occurrences != 1:
        raise B1RCellError(f"DFL optimizer occurrence count is {occurrences}, expected 1")
    for group in optimizer.param_groups:
        group["params"] = [candidate for candidate in group["params"] if candidate is not parameter]
    after = _optimizer_parameters(optimizer)
    if len(after) != len(before) - 1 or any(candidate is parameter for candidate in after):
        raise B1RCellError("DFL exclusion postcondition failed")
    validate_optimizer_coverage(model, optimizer)
    return {
        "schema": "A1_PHASEB1R_DFL_OPTIMIZER_EXCLUSION_V1",
        "source": "ULTRALYTICS_STANDARD_BUILD_OPTIMIZER",
        "timing": "AFTER_STANDARD_BUILD_BEFORE_TRAINING",
        "removed_name": name,
        "removed_shape": list(shape),
        "removed_numel": LOCKED_DFL_NUMEL,
        "removed_count": 1,
        "optimizer_groups_before": LOCKED_OPTIMIZER_GROUPS,
        "optimizer_groups_after": LOCKED_OPTIMIZER_GROUPS,
        "postcondition": "ALL_AND_ONLY_REQUIRES_GRAD",
    }


def validate_optimizer_coverage(model: Any, optimizer: Any) -> dict[str, int]:
    try:
        parameters = list(model.parameters())
    except (AttributeError, TypeError) as exc:
        raise B1RCellError("model parameters surface is absent") from exc
    trainable = [parameter for parameter in parameters if getattr(parameter, "requires_grad", None) is True]
    observed = _optimizer_parameters(optimizer)
    trainable_ids = [id(parameter) for parameter in trainable]
    observed_ids = [id(parameter) for parameter in observed]
    if len(trainable_ids) != len(set(trainable_ids)) or len(observed_ids) != len(set(observed_ids)):
        raise B1RCellError("duplicate parameter identity")
    if set(trainable_ids) != set(observed_ids):
        raise B1RCellError(
            f"optimizer coverage mismatch: missing={len(set(trainable_ids)-set(observed_ids))}, "
            f"extra={len(set(observed_ids)-set(trainable_ids))}"
        )
    return {
        "model_parameters": len(parameters),
        "trainable_parameters": len(trainable),
        "optimizer_parameters": len(observed),
        "optimizer_groups": LOCKED_OPTIMIZER_GROUPS,
    }


def validate_batch_surface(
    *,
    args_batch: Any,
    trainer_batch: Any,
    loader_batch: Any,
    image_batch: Any,
    allow_final_partial_image_batch: bool = False,
) -> None:
    configured = (args_batch, trainer_batch, loader_batch)
    if any(type(value) is not int for value in configured) or any(value != 2 for value in configured):
        raise B1RCellError(f"locked configured batch changed: {configured}")
    allowed_image_batches = {1, 2} if allow_final_partial_image_batch else {2}
    if type(image_batch) is not int or image_batch not in allowed_image_batches:
        raise B1RCellError(f"invalid image batch: {image_batch}")


def build_trainer_class(stack: Mapping[str, Any]) -> type:
    """Build the minimal controlled subclass from an attested numeric stack."""

    required = {"DetectionTrainer", "build_dataloader", "data_adapter", "torch", "np"}
    if set(stack) != required:
        raise B1RCellError("numeric stack keys are not exact")
    base = stack["DetectionTrainer"]
    torch_module = stack["torch"]
    np_module = stack["np"]
    build_dataloader = stack["build_dataloader"]
    data_adapter = stack["data_adapter"]
    if not isinstance(base, type) or not callable(build_dataloader):
        raise B1RCellError("numeric stack trainer/dataloader surface is absent")

    class PhaseB1RTrainer(base):
        """Ultralytics loop with only the B1-R boundary overrides."""

        def __init__(self, *, context: Mapping[str, Any], overrides: Mapping[str, Any], callbacks: Any = None):
            if set(context) != {
                "cell_id",
                "model",
                "image_paths",
                "data",
                "cell_root",
                "recovery_checkpoint",
                "checkpoint_path",
                "fresh_model_factory",
            }:
                raise B1RCellError("trainer context keys are not exact")
            self.phaseb1r_context = dict(context)
            self.phaseb1r_optimizer_report = None
            self.phaseb1r_optimizer_step_calls = 0
            self.phaseb1r_amp_skips = 0
            self.phaseb1r_resumed_from_epoch = None
            self.phaseb1r_final_checkpoint_written = False
            self.phaseb1r_checkpoint_qc = None
            if not isinstance(callbacks, dict) or not callbacks or any(not isinstance(value, list) for value in callbacks.values()):
                raise B1RCellError("an explicit nonempty local callback table is required")
            inherited_init = super().__init__
            callback_modules = []
            for ancestor in base.__mro__:
                ancestor_init = ancestor.__dict__.get("__init__")
                init_globals = getattr(ancestor_init, "__globals__", None)
                candidate = init_globals.get("callbacks") if isinstance(init_globals, dict) else None
                if callable(getattr(candidate, "add_integration_callbacks", None)):
                    callback_modules.append(candidate)
            callback_modules = list(dict.fromkeys(callback_modules))
            callbacks_module = callback_modules[0] if len(callback_modules) == 1 else None
            add_integrations = getattr(callbacks_module, "add_integration_callbacks", None)
            if not callable(add_integrations):
                raise B1RCellError("upstream integration callback surface changed")

            def integrations_forbidden(_trainer):
                return None

            callbacks_module.add_integration_callbacks = integrations_forbidden
            try:
                inherited_init(overrides=dict(overrides), _callbacks=callbacks)
            finally:
                callbacks_module.add_integration_callbacks = add_integrations
            self.model = context["model"]

        def get_dataset(self):
            data = self.phaseb1r_context["data"]
            if not isinstance(data, Mapping) or set(data) != {"names", "nc", "channels"}:
                raise B1RCellError("manifest-bound data document keys are not exact")
            if data["nc"] != 10 or len(data["names"]) != 10 or data["channels"] != 3:
                raise B1RCellError("manifest-bound data document differs from VisDrone-10")
            return {**dict(data), "train": "PHASEB1R_TRAIN_MANIFEST", "val": None}

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train" or dataset_path != "PHASEB1R_TRAIN_MANIFEST" or rank not in {-1, 0}:
                raise B1RCellError("only the manifest-bound single-GPU train loader is allowed")
            if type(batch_size) is not int or batch_size != 2:
                raise B1RCellError("train loader batch is not exact integer 2")
            image_paths = list(self.phaseb1r_context["image_paths"])
            identities = {os.path.normcase(str(Path(path).resolve())) for path in image_paths}
            if len(image_paths) != 5149 or len(identities) != 5149:
                raise B1RCellError("train manifest image closure is not exactly 5,149 unique paths")
            original_np_load = np_module.load

            def forbidden_np_load(*_args, **_kwargs):
                raise B1RCellError("np.load/.npy input is forbidden")

            np_module.load = forbidden_np_load
            try:
                dataset = data_adapter.build_read_only_jpeg_dataset(
                    cfg=self.args,
                    image_paths=[Path(path).resolve() for path in image_paths],
                    data=dict(self.phaseb1r_context["data"]),
                    batch_size=2,
                    augment=True,
                    rect=False,
                    stride=32,
                )
            finally:
                np_module.load = original_np_load
            loader = build_dataloader(
                dataset,
                batch=2,
                workers=0,
                shuffle=True,
                rank=-1,
                drop_last=False,
                pin_memory=True,
            )
            validate_batch_surface(
                args_batch=self.args.batch,
                trainer_batch=self.batch_size,
                loader_batch=loader.batch_size,
                image_batch=2,
            )
            return loader

        def _setup_train(self):
            inherited = super()._setup_train
            function = getattr(inherited, "__func__", None)
            globals_map = getattr(function, "__globals__", None)
            if not isinstance(globals_map, dict) or "check_amp" not in globals_map:
                raise B1RCellError("upstream AMP check hook surface changed")
            original_check_amp = globals_map["check_amp"]

            def locked_amp_attestation(model):
                if model is not self.model or self.args.amp is not True:
                    raise B1RCellError("AMP attestation received the wrong model/config")
                return True

            globals_map["check_amp"] = locked_amp_attestation
            try:
                inherited()
            finally:
                globals_map["check_amp"] = original_check_amp
            if self.amp is not True:
                raise B1RCellError("formal AMP was not retained after setup")

            recovery = self.phaseb1r_context["recovery_checkpoint"]
            if recovery is not None:
                self._restore_recovery_checkpoint(Path(recovery))

        def _restore_recovery_checkpoint(self, checkpoint: Path):
            cell_root = Path(self.phaseb1r_context["cell_root"]).resolve()
            checkpoint = checkpoint.resolve()
            try:
                checkpoint.relative_to(cell_root)
            except ValueError as exc:
                raise B1RCellError("recovery checkpoint escapes its cell directory") from exc
            if checkpoint.name != "last.pt" or checkpoint.parent.name != "weights":
                raise B1RCellError("recovery checkpoint must be the cell's standard weights/last.pt")
            if not checkpoint.is_file() or checkpoint.is_symlink():
                raise B1RCellError("recovery checkpoint is missing/non-regular")
            payload = torch_module.load(checkpoint, map_location=self.device, weights_only=False)
            if not isinstance(payload, Mapping):
                raise B1RCellError("recovery checkpoint root is not a mapping")
            epoch = payload.get("epoch")
            if type(epoch) is not int or not 0 <= epoch <= 149:
                raise B1RCellError("recovery checkpoint epoch is outside 0..149")
            train_args = payload.get("train_args")
            if not isinstance(train_args, Mapping):
                raise B1RCellError("recovery checkpoint lacks train_args")
            expected = {
                "seed": 42,
                "batch": 2,
                "epochs": 150,
                "optimizer": "SGD",
                "val": False,
                "save_period": 5,
            }
            changed = [key for key, value in expected.items() if train_args.get(key) != value]
            observed_imgsz = train_args.get("imgsz")
            if observed_imgsz not in (768, [768, 768], (768, 768)):
                changed.append("imgsz")
            if changed:
                raise B1RCellError(f"recovery checkpoint training config changed: {sorted(set(changed))}")
            ema_model = payload.get("ema")
            if ema_model is None or not hasattr(ema_model, "float") or not hasattr(ema_model, "state_dict"):
                raise B1RCellError("recovery checkpoint lacks the standard EMA model")
            result = self.model.load_state_dict(ema_model.float().state_dict(), strict=True)
            if list(getattr(result, "missing_keys", [])) or list(getattr(result, "unexpected_keys", [])):
                raise B1RCellError("recovery EMA strict model load failed")
            self._load_checkpoint_state(payload)
            self.start_epoch = epoch + 1
            self.scheduler.last_epoch = self.start_epoch - 1
            if self.start_epoch > self.epochs - self.args.close_mosaic:
                self._close_dataloader_mosaic()
            self.phaseb1r_resumed_from_epoch = epoch

        def _build_train_pipeline(self):
            validate_batch_surface(
                args_batch=self.args.batch,
                trainer_batch=self.batch_size,
                loader_batch=2,
                image_batch=2,
            )
            self.train_loader = self.get_dataloader(self.data["train"], batch_size=2, rank=-1, mode="train")
            self.test_loader = None
            self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
            weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
            iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
            self.optimizer = self.build_optimizer(
                model=self.model,
                name=self.args.optimizer,
                lr=self.args.lr0,
                momentum=self.args.momentum,
                decay=weight_decay,
                iterations=iterations,
            )
            self._setup_scheduler()

        def get_validator(self):
            return SimpleNamespace(metrics=SimpleNamespace(keys=[]))

        def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
            if name != "SGD":
                raise B1RCellError("formal optimizer must be SGD")
            optimizer = super().build_optimizer(
                model=model,
                name=name,
                lr=lr,
                momentum=momentum,
                decay=decay,
                iterations=iterations,
            )
            exclusion = exclude_locked_dfl_after_standard_build(model, optimizer)
            coverage = validate_optimizer_coverage(model, optimizer)
            self.phaseb1r_optimizer_report = {**coverage, "dfl_exclusion": exclusion}

            def count_step(_optimizer, _args, _kwargs):
                self.phaseb1r_optimizer_step_calls += 1

            self.phaseb1r_optimizer_hook = optimizer.register_step_post_hook(count_step)
            return optimizer

        def preprocess_batch(self, batch):
            processed = super().preprocess_batch(batch)
            validate_batch_surface(
                args_batch=self.args.batch,
                trainer_batch=self.batch_size,
                loader_batch=self.train_loader.batch_size,
                image_batch=int(processed["img"].shape[0]),
                allow_final_partial_image_batch=True,
            )
            return processed

        def optimizer_step(self):
            if self.phaseb1r_optimizer_report is None:
                raise B1RCellError("optimizer QC report is absent")
            self.scaler.unscale_(self.optimizer)
            trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            gradients = [parameter for parameter in trainable if parameter.grad is not None]
            if not gradients:
                raise B1RCellError("optimizer step has no gradients")
            gradients_finite = all(bool(torch_module.isfinite(parameter.grad).all()) for parameter in gradients)
            if gradients_finite:
                torch_module.nn.utils.clip_grad_norm_(gradients, max_norm=10.0, error_if_nonfinite=True)
            calls_before = self.phaseb1r_optimizer_step_calls
            scale_before = float(self.scaler.get_scale())
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = float(self.scaler.get_scale())
            calls_after = self.phaseb1r_optimizer_step_calls
            normal_step = calls_after == calls_before + 1 and scale_after >= scale_before
            amp_skip = calls_after == calls_before and scale_after < scale_before
            if gradients_finite and not normal_step:
                raise B1RCellError("finite-gradient optimizer step did not complete exactly once")
            if not gradients_finite and not amp_skip:
                raise B1RCellError("non-finite gradients were not handled by one standard AMP skip")
            if amp_skip:
                self.phaseb1r_amp_skips += 1
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema and not amp_skip:
                self.ema.update(self.model)

        def validate(self):
            return {}, None

        def _handle_nan_recovery(self, epoch):
            if self.loss is not None and not bool(torch_module.isfinite(self.loss)):
                raise B1RCellError(f"non-finite loss at epoch {epoch}; rerun the same command to resume last.pt")
            return False

        def save_model(self):
            if self.epochs != 150 or not 0 <= self.epoch <= 149:
                raise B1RCellError("checkpoint save attempted outside the fixed 150-epoch run")
            if (self.epoch + 1) % 5 != 0:
                return False
            standard_saved = super().save_model()
            if standard_saved is not True:
                raise B1RCellError("standard Ultralytics recovery checkpoint was not saved")
            if self.epoch != 149:
                return True
            self.publish_final_checkpoint()
            return True

        def publish_final_checkpoint(self):
            current_epoch = getattr(self, "epoch", None)
            if current_epoch != 149 and self.phaseb1r_resumed_from_epoch != 149:
                raise B1RCellError("final checkpoint publication is not at epoch 149")
            target = Path(self.phaseb1r_context["checkpoint_path"]).resolve()
            if target.name != contract.expected_checkpoint_name(self.phaseb1r_context["cell_id"]):
                raise B1RCellError("final checkpoint path/name mismatch")
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise B1RCellError("final checkpoint target is not a regular file")
            temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
            if temporary.exists() or temporary.is_symlink():
                raise B1RCellError("checkpoint temporary path already exists")
            state = {name: value.detach().cpu() for name, value in self.ema.ema.state_dict().items()}
            state_sha256 = state_mapping_sha256(torch_module, state)
            payload = {
                "schema": "A1_PHASEB1R_FINAL_EMA_CHECKPOINT_V1",
                "cell_id": self.phaseb1r_context["cell_id"],
                "epoch_index": 149,
                "ema_updates": int(self.ema.updates),
                "ema_state_sha256": state_sha256,
                "ema_state_dict": state,
            }
            if not target.exists():
                try:
                    torch_module.save(payload, temporary)
                    with temporary.open("r+b") as stream:
                        os.fsync(stream.fileno())
                    if target.exists() or target.is_symlink():
                        raise B1RCellError("checkpoint target appeared during publication")
                    os.replace(temporary, target)
                finally:
                    if temporary.exists():
                        temporary.unlink()

            # Strict reload QC: load the published payload into a fresh same-cell
            # model graph before any training COMMIT may be emitted.
            loaded = torch_module.load(target, map_location="cpu", weights_only=True)
            required_keys = {"schema", "cell_id", "epoch_index", "ema_updates", "ema_state_sha256", "ema_state_dict"}
            if not isinstance(loaded, dict) or set(loaded) != required_keys:
                raise B1RCellError("checkpoint payload keys are not exact")
            if loaded["schema"] != payload["schema"] or loaded["cell_id"] != payload["cell_id"] or loaded["epoch_index"] != 149:
                raise B1RCellError("checkpoint identity metadata mismatch after reload")
            if loaded["ema_state_sha256"] != state_sha256:
                raise B1RCellError("checkpoint state hash metadata mismatch")
            if state_mapping_sha256(torch_module, loaded["ema_state_dict"]) != state_sha256:
                raise B1RCellError("checkpoint serialized state hash mismatch")
            fresh_factory = self.phaseb1r_context["fresh_model_factory"]
            if not callable(fresh_factory):
                raise B1RCellError("fresh same-cell model factory is absent")
            fresh_model = fresh_factory()
            if fresh_model is None or not hasattr(fresh_model, "state_dict") or not hasattr(fresh_model, "load_state_dict"):
                raise B1RCellError("fresh same-cell model surface is absent")
            if hasattr(fresh_model, "cpu"):
                fresh_model = fresh_model.cpu()
            initial_sha256 = model_state_sha256(torch_module, fresh_model)
            if initial_sha256 != contract.INITIAL_STATE_SHA256[self.phaseb1r_context["cell_id"]]:
                raise B1RCellError("fresh reload graph does not match the B0 initial state")
            result = fresh_model.load_state_dict(loaded["ema_state_dict"], strict=True)
            missing = list(getattr(result, "missing_keys", []))
            unexpected = list(getattr(result, "unexpected_keys", []))
            if missing or unexpected:
                raise B1RCellError(f"strict checkpoint reload coverage failed: missing={missing}, unexpected={unexpected}")
            reloaded_sha256 = model_state_sha256(torch_module, fresh_model)
            if reloaded_sha256 != state_sha256:
                raise B1RCellError("fresh-model reload state hash mismatch")
            self.phaseb1r_checkpoint_qc = {
                "schema": "A1_PHASEB1R_CHECKPOINT_QC_V1",
                "path": str(target),
                "bytes": int(target.stat().st_size),
                "file_sha256": contract.file_sha256(target),
                "ema_state_sha256": state_sha256,
                "fresh_model_initial_state_sha256": initial_sha256,
                "fresh_model_reloaded_state_sha256": reloaded_sha256,
                "strict": True,
                "passed": True,
            }
            self.phaseb1r_final_checkpoint_written = True
            return dict(self.phaseb1r_checkpoint_qc)

        def final_eval(self):
            if not self.phaseb1r_final_checkpoint_written:
                raise B1RCellError("training ended without the epoch-149 final EMA checkpoint")

    PhaseB1RTrainer.__name__ = "PhaseB1RTrainer"
    return PhaseB1RTrainer
def main() -> None:
    raise B1RCellError(CELL_ENGINE_BLOCK_REASON)


if __name__ == "__main__":
    main()
