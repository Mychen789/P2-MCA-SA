from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import ultralytics
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import initialize_weights


SCHEMA = "A1_PHASEB0_BUILDER_V1"
NC = 10
REG_MAX = 16
COMMON_TRANSFER_MAX_LAYER = 15
TRUSTED_WEIGHTS_SHA256 = "5D4A90CDC7A21786CC59CD19778E9EAFFF836DF9E2DA32524737C7EE6EFE4FE5"
SEMANTIC_BODY_PAIRS: tuple[tuple[int, int], ...] = ((16, 22), (18, 24), (19, 25), (21, 27))


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    p2: bool
    c3: int
    expected_total_params: int


CELL_SPECS: tuple[CellSpec, ...] = (
    CellSpec("C00", False, 96, 24_116_254),
    CellSpec("C01", False, 192, 25_862_110),
    CellSpec("C10", True, 96, 25_055_336),
    CellSpec("C11", True, 192, 27_134_312),
)


@dataclass
class BuiltCell:
    spec: CellSpec
    model: DetectionModel
    build_report: dict[str, Any]
    transfer_rows: list[dict[str, Any]]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest().upper()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = canonical_bytes({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}) + b"\0"
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + raw).hexdigest().upper()


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return int(sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)))


def _derive_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _validate_legacy_tower(tower: nn.Module, expected_in: int, expected_width: int, outputs: int) -> None:
    if not isinstance(tower, nn.Sequential) or len(tower) != 3:
        raise RuntimeError("fixed head requires legacy Sequential[Conv, Conv, Conv2d]")
    if not isinstance(tower[0], Conv) or not isinstance(tower[1], Conv) or not isinstance(tower[2], nn.Conv2d):
        raise RuntimeError("classification/regression tower is not legacy Conv-Conv-Conv2d")
    observed = (
        int(tower[0].conv.in_channels),
        int(tower[0].conv.out_channels),
        int(tower[1].conv.in_channels),
        int(tower[1].conv.out_channels),
        int(tower[2].in_channels),
        int(tower[2].out_channels),
    )
    expected = (expected_in, expected_width, expected_width, expected_width, expected_width, outputs)
    if observed != expected:
        raise RuntimeError(f"legacy tower mismatch: observed={observed}, expected={expected}")
    for block in tower[:2]:
        if not isinstance(block.bn, nn.BatchNorm2d) or block.bn.eps != 1e-3 or block.bn.momentum != 0.03:
            raise RuntimeError("legacy tower BN hyperparameters do not match Ultralytics initialize_weights")


def set_fixed_legacy_cv3(model: DetectionModel, c3: int) -> dict[str, Any]:
    if type(c3) is not int or c3 <= 0:
        raise ValueError("c3 must be a positive exact int")
    detect = model.model[-1]
    if not isinstance(detect, Detect) or detect.end2end:
        raise RuntimeError("Phase-B0 requires one non-end2end Detect head")
    inputs = [int(tower[0].conv.in_channels) for tower in detect.cv3]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    replacement = nn.ModuleList(
        nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, int(detect.nc), 1)) for x in inputs
    ).to(device=device, dtype=dtype)
    detect.cv3 = replacement
    initialize_weights(detect.cv3)
    for tower, x in zip(detect.cv3, inputs):
        _validate_legacy_tower(tower, x, c3, int(detect.nc))
    return {"inputs": inputs, "c3": c3, "nc": int(detect.nc), "implementation": "legacy_Conv_Conv_Conv2d"}


def _reset_tower(tower: nn.Module, seed: int) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module in tower.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                module.reset_parameters()
        initialize_weights(tower)


def reset_detect_heads(model: DetectionModel, training_seed: int, c3: int) -> dict[str, Any]:
    """Reset all cv2/cv3 towers by stable stride identity, then apply upstream Detect bias initialization."""
    detect = model.model[-1]
    if not isinstance(detect, Detect) or detect.end2end:
        raise RuntimeError("unexpected Detect head")
    strides = [int(float(x)) for x in detect.stride]
    if any(stride <= 0 for stride in strides):
        raise RuntimeError(f"Detect strides are not initialized: {strides}")
    rows: list[dict[str, Any]] = []
    for index, stride in enumerate(strides):
        for tower_type, tower, width_tag in (
            ("cv2", detect.cv2[index], REG_MAX),
            ("cv3", detect.cv3[index], c3),
        ):
            seed = _derive_seed("A1_PHASEB_HEAD_INIT_V1", training_seed, stride, tower_type, width_tag)
            _reset_tower(tower, seed)
            rows.append({"stride": stride, "tower_type": tower_type, "width_tag": width_tag, "seed": seed})
    detect.bias_init()
    bias_rows: list[dict[str, Any]] = []
    for index, stride in enumerate(strides):
        box_bias = detect.cv2[index][-1].bias.detach()
        class_bias = detect.cv3[index][-1].bias.detach()
        expected_class = math.log(5 / int(detect.nc) / (640 / stride) ** 2)
        if not torch.equal(box_bias, torch.full_like(box_bias, 2.0)):
            raise RuntimeError(f"stride {stride}: Detect box bias initialization mismatch")
        if not torch.allclose(
            class_bias,
            torch.full_like(class_bias, expected_class),
            rtol=0.0,
            atol=1e-7,
        ):
            raise RuntimeError(f"stride {stride}: Detect class bias initialization mismatch")
        bias_rows.append(
            {
                "stride": stride,
                "box_bias_sha256": tensor_sha256(box_bias),
                "class_bias_sha256": tensor_sha256(class_bias),
                "expected_class_bias": expected_class,
            }
        )
    return {
        "schema": "A1_PHASEB_HEAD_INIT_V1",
        "training_seed": int(training_seed),
        "rows": rows,
        "bias_qc": bias_rows,
        "batchnorm": {"eps": 1e-3, "momentum": 0.03},
    }


def synchronize_semantic_body_initialization(cells: dict[str, BuiltCell]) -> dict[str, Any]:
    """Make shared P3/P4/P5 bottom-up modules exact across the P2 factor.

    This is random-initialization synchronization, not an expansion of the
    COCO transfer whitelist.  Modules unique to the P2 path remain independent.
    """
    rows: list[dict[str, Any]] = []
    for c3, off_id, on_id in ((96, "C00", "C10"), (192, "C01", "C11")):
        off_model, on_model = cells[off_id].model, cells[on_id].model
        for off_index, on_index in SEMANTIC_BODY_PAIRS:
            source = off_model.model[off_index]
            target = on_model.model[on_index]
            source_state, target_state = source.state_dict(), target.state_dict()
            if list(source_state) != list(target_state):
                raise RuntimeError(f"semantic module key mismatch: {off_id}.{off_index}/{on_id}.{on_index}")
            if any(source_state[key].shape != target_state[key].shape for key in source_state):
                raise RuntimeError(f"semantic module shape mismatch: {off_id}.{off_index}/{on_id}.{on_index}")
            target.load_state_dict(source_state, strict=True)
            source_rows = _state_spec(source, include_hash=True)
            target_rows = _state_spec(target, include_hash=True)
            exact = source_rows == target_rows
            if not exact:
                raise RuntimeError(f"semantic module state mismatch: {off_id}.{off_index}/{on_id}.{on_index}")
            rows.append(
                {
                    "c3": c3,
                    "off": f"{off_id}.model.{off_index}",
                    "on": f"{on_id}.model.{on_index}",
                    "state_exact": exact,
                    "role": "SYNCHRONIZED_RANDOM_INIT_NOT_PRETRAINED",
                    "state_sha256": canonical_sha256(source_rows),
                }
            )
    return {"schema": "A1_PHASEB_SEMANTIC_BODY_INIT_V1", "pairs": rows}


def verify_semantic_body_initialization(cells: dict[str, BuiltCell]) -> dict[str, Any]:
    """Pure verification of the already synchronized same-role random modules."""
    rows: list[dict[str, Any]] = []
    for c3, off_id, on_id in ((96, "C00", "C10"), (192, "C01", "C11")):
        off_model, on_model = cells[off_id].model, cells[on_id].model
        for off_index, on_index in SEMANTIC_BODY_PAIRS:
            source_rows = _state_spec(off_model.model[off_index], include_hash=True)
            target_rows = _state_spec(on_model.model[on_index], include_hash=True)
            exact = source_rows == target_rows
            rows.append(
                {
                    "c3": c3,
                    "off": f"{off_id}.model.{off_index}",
                    "on": f"{on_id}.model.{on_index}",
                    "state_exact": exact,
                    "role": "VERIFIED_SYNCHRONIZED_RANDOM_INIT_NOT_PRETRAINED",
                    "state_sha256": canonical_sha256(source_rows),
                }
            )
            if not exact:
                raise RuntimeError(f"semantic body state mismatch: {off_id}.{off_index}/{on_id}.{on_index}")
    return {"schema": "A1_PHASEB_SEMANTIC_BODY_INIT_QC_V1", "pairs": rows, "verification_mutated_model": False}


def _state_spec(model: nn.Module, predicate=lambda _name: True, include_hash: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, value in model.state_dict().items():
        if predicate(name):
            row = {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
            if include_hash:
                row["sha256"] = tensor_sha256(value)
            rows.append(row)
    return rows


def _module_layer(name: str) -> int | None:
    match = re.match(r"^model\.(\d+)(?:\.|$)", name)
    return None if match is None else int(match.group(1))


def is_common_transfer_key(name: str) -> bool:
    layer = _module_layer(name)
    return layer is not None and layer <= COMMON_TRANSFER_MAX_LAYER


def _head_tower_state(detect: Detect, tower_type: str, stride: int) -> list[dict[str, Any]]:
    strides = [int(float(x)) for x in detect.stride]
    if stride not in strides:
        raise KeyError(f"stride {stride} absent from head {strides}")
    index = strides.index(stride)
    tower = getattr(detect, tower_type)[index]
    return [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha256(value)}
        for name, value in tower.state_dict().items()
    ]


def build_cell(
    spec: CellSpec,
    p2_yaml: Path,
    training_seed: int = 42,
    official_yaml: Path | None = None,
) -> BuiltCell:
    p2_yaml = p2_yaml.resolve()
    if official_yaml is None:
        official_yaml = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "v8" / "yolov8.yaml"
    official_yaml = official_yaml.resolve()
    if not p2_yaml.is_file() or not official_yaml.is_file():
        raise FileNotFoundError("locked P2/official YAML is missing")
    with torch.random.fork_rng(devices=[]):
        # Cells that differ only in c3 must share every pre-override random draw.
        torch.manual_seed(_derive_seed("A1_PHASEB_GRAPH_BUILD_V1", "P2_ON" if spec.p2 else "P2_OFF", training_seed))
        if spec.p2:
            p2_cfg = dict(YAML.load(p2_yaml))
            p2_cfg["scale"] = "m"
            p2_cfg["yaml_file"] = str(p2_yaml)
            model = DetectionModel(p2_cfg, ch=3, nc=NC, verbose=False)
            config_role = "workspace_yolov8m_p2"
            config_path = p2_yaml
        else:
            official_cfg = dict(YAML.load(official_yaml))
            official_cfg["scale"] = "m"
            official_cfg["yaml_file"] = str(official_yaml)
            model = DetectionModel(official_cfg, ch=3, nc=NC, verbose=False)
            config_role = "installed_official_yolov8m"
            config_path = official_yaml
        if model.yaml.get("scale") != "m":
            raise RuntimeError(f"{spec.cell_id}: locked model scale is not m: {model.yaml.get('scale')}")
        override = set_fixed_legacy_cv3(model, spec.c3)
        init_report = reset_detect_heads(model, training_seed, spec.c3)

    detect = model.model[-1]
    if not isinstance(detect, Detect):
        raise RuntimeError(f"{spec.cell_id}: final module is not Detect")
    strides = [int(float(x)) for x in detect.stride]
    expected_strides = [4, 8, 16, 32] if spec.p2 else [8, 16, 32]
    if strides != expected_strides:
        raise RuntimeError(f"{spec.cell_id}: strides {strides} != {expected_strides}")
    widths = [int(tower[0].conv.out_channels) for tower in detect.cv3]
    if widths != [spec.c3] * len(widths):
        raise RuntimeError(f"{spec.cell_id}: cv3 widths {widths}")
    total = count_parameters(model)
    if total != spec.expected_total_params:
        raise RuntimeError(f"{spec.cell_id}: total params {total} != {spec.expected_total_params}")
    if int(detect.nc) != NC or int(detect.reg_max) != REG_MAX or int(detect.no) != NC + 4 * REG_MAX:
        raise RuntimeError(f"{spec.cell_id}: Detect nc/reg_max/no mismatch")
    expected_inputs = [96, 192, 384, 576] if spec.p2 else [192, 384, 576]
    expected_from = [18, 21, 24, 27] if spec.p2 else [15, 18, 21]
    cv2_widths = [int(tower[0].conv.out_channels) for tower in detect.cv2]
    detect_from = [int(value) for value in detect.f]
    if (
        override["inputs"] != expected_inputs
        or int(detect.nl) != len(expected_strides)
        or cv2_widths != [64] * len(expected_strides)
        or detect_from != expected_from
    ):
        raise RuntimeError(
            f"{spec.cell_id}: Detect wiring mismatch inputs={override['inputs']} nl={detect.nl} "
            f"cv2={cv2_widths} from={detect_from}"
        )

    report = {
        "schema": SCHEMA,
        "cell_id": spec.cell_id,
        "p2": spec.p2,
        "c3": spec.c3,
        "nc": int(detect.nc),
        "reg_max": int(detect.reg_max),
        "no": int(detect.no),
        "strides": strides,
        "detect_inputs": override["inputs"],
        "detect_from": detect_from,
        "detect_nl": int(detect.nl),
        "cv2_hidden_widths": cv2_widths,
        "cv3_hidden_widths": widths,
        "params_total": total,
        "params_trainable": count_parameters(model, True),
        "params_cv2": count_parameters(detect.cv2),
        "params_cv3": count_parameters(detect.cv3),
        "config_role": config_role,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "head_init": init_report,
        "common_transfer_state_spec": _state_spec(model, is_common_transfer_key, include_hash=False),
        "non_cv3_state_spec": _state_spec(model, lambda name: ".cv3." not in name, include_hash=False),
    }
    return BuiltCell(spec=spec, model=model, build_report=report, transfer_rows=[])


def build_cells(
    p2_yaml: Path,
    training_seed: int = 42,
    official_yaml: Path | None = None,
) -> dict[str, BuiltCell]:
    cells = {
        spec.cell_id: build_cell(spec, p2_yaml, training_seed, official_yaml=official_yaml)
        for spec in CELL_SPECS
    }
    baseline_spec = cells["C00"].build_report["common_transfer_state_spec"]
    for cell in cells.values():
        if cell.build_report["common_transfer_state_spec"] != baseline_spec:
            raise RuntimeError(f"{cell.spec.cell_id}: common model.0-15 state spec differs")
    semantic_report = synchronize_semantic_body_initialization(cells)
    for cell in cells.values():
        cell.build_report["semantic_body_initialization"] = semantic_report
    return cells


def transfer_common_pretrained(cells: dict[str, BuiltCell], weights_path: Path) -> dict[str, Any]:
    observed_weights_sha256 = file_sha256(weights_path)
    if observed_weights_sha256 != TRUSTED_WEIGHTS_SHA256:
        raise RuntimeError(
            f"untrusted pretrained container: {observed_weights_sha256} != {TRUSTED_WEIGHTS_SHA256}"
        )
    holder = YOLO(str(weights_path))
    source = holder.model.cpu()
    source_state = source.state_dict()
    source_keys = sorted(name for name in source_state if is_common_transfer_key(name))
    if not source_keys:
        raise RuntimeError("pretrained source has no model.0-15 state")

    reference_target_keys: list[str] | None = None
    for cell in cells.values():
        target_state = cell.model.state_dict()
        target_keys = sorted(name for name in target_state if is_common_transfer_key(name))
        if reference_target_keys is None:
            reference_target_keys = target_keys
        if target_keys != reference_target_keys or target_keys != source_keys:
            raise RuntimeError(f"{cell.spec.cell_id}: transfer whitelist key set mismatch")
        before_all = {name: tensor_sha256(value) for name, value in target_state.items()}
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for name in source_keys:
                src = source_state[name].detach().cpu()
                dst = target_state[name]
                if src.shape != dst.shape or src.dtype != dst.dtype:
                    raise RuntimeError(f"{cell.spec.cell_id}: incompatible transfer tensor {name}")
                before = tensor_sha256(dst)
                dst.copy_(src.to(device=dst.device))
                after = tensor_sha256(dst)
                source_hash = tensor_sha256(src)
                if after != source_hash:
                    raise RuntimeError(f"{cell.spec.cell_id}: transfer hash mismatch {name}")
                rows.append(
                    {
                        "name": name,
                        "source_key": name,
                        "target_key": name,
                        "shape": list(dst.shape),
                        "dtype": str(dst.dtype),
                        "before_sha256": before,
                        "source_sha256": source_hash,
                        "after_sha256": after,
                        "status": "LOADED_WHITELIST",
                        "reason": "EXACT_MODEL_0_TO_15_WHITELIST",
                    }
                )
        after_all = {name: tensor_sha256(value) for name, value in target_state.items()}
        changed_outside = [
            name for name in target_state if not is_common_transfer_key(name) and before_all[name] != after_all[name]
        ]
        if changed_outside:
            raise RuntimeError(f"{cell.spec.cell_id}: transfer changed non-whitelist tensors {changed_outside[:5]}")
        for name in sorted(target_state):
            if is_common_transfer_key(name):
                continue
            target = target_state[name]
            source_value = source_state.get(name)
            rows.append(
                {
                    "name": name,
                    "source_key": name if source_value is not None else None,
                    "target_key": name,
                    "target_shape": list(target.shape),
                    "target_dtype": str(target.dtype),
                    "target_sha256": tensor_sha256(target),
                    "source_exists": source_value is not None,
                    "source_shape": None if source_value is None else list(source_value.shape),
                    "source_dtype": None if source_value is None else str(source_value.dtype),
                    "source_sha256": None if source_value is None else tensor_sha256(source_value),
                    "status": "SKIPPED_NON_WHITELIST",
                    "reason": "TARGET_LAYER_OUTSIDE_MODEL_0_TO_15",
                }
            )
        cell.transfer_rows = rows

    reference_cell_id = sorted(cells)[0]
    trunk_reference = [
        {"name": name, "sha256": tensor_sha256(cells[reference_cell_id].model.state_dict()[name])}
        for name in source_keys
    ]
    for cell in cells.values():
        candidate = [{"name": name, "sha256": tensor_sha256(cell.model.state_dict()[name])} for name in source_keys]
        if candidate != trunk_reference:
            raise RuntimeError(f"{cell.spec.cell_id}: transferred common trunk differs")

    del holder, source
    return {
        "schema": "A1_PHASEB_COMMON_TRANSFER_V1",
        "weights_path": str(weights_path),
        "weights_sha256": observed_weights_sha256,
        "allowed_layer_range": [0, COMMON_TRANSFER_MAX_LAYER],
        "keys_loaded_per_cell": len(source_keys),
        "rows_per_cell": {cell_id: len(cell.transfer_rows) for cell_id, cell in sorted(cells.items())},
        "skipped_non_whitelist_per_cell": {
            cell_id: sum(row["status"] == "SKIPPED_NON_WHITELIST" for row in cell.transfer_rows)
            for cell_id, cell in sorted(cells.items())
        },
        "key_set_sha256": canonical_sha256(source_keys),
        "common_trunk_state_sha256": canonical_sha256(trunk_reference),
    }


def verify_pair_invariance(cells: dict[str, BuiltCell]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for low_id, high_id in (("C00", "C01"), ("C10", "C11")):
        low, high = cells[low_id], cells[high_id]
        low_state = _state_spec(low.model, lambda name: ".cv3." not in name, include_hash=True)
        high_state = _state_spec(high.model, lambda name: ".cv3." not in name, include_hash=True)
        exact = low_state == high_state
        rows.append({"pair": f"{low_id}:{high_id}", "fixed_p2": low.spec.p2, "non_cv3_state_exact": exact})
        if not exact:
            raise RuntimeError(f"{low_id}/{high_id}: non-cv3 state differs")

    common_head_rows: list[dict[str, Any]] = []
    for c3, off_id, on_id in ((96, "C00", "C10"), (192, "C01", "C11")):
        off_detect = cells[off_id].model.model[-1]
        on_detect = cells[on_id].model.model[-1]
        for stride in (8, 16, 32):
            for tower_type in ("cv2", "cv3"):
                left = _head_tower_state(off_detect, tower_type, stride)
                right = _head_tower_state(on_detect, tower_type, stride)
                exact = left == right
                common_head_rows.append(
                    {"c3": c3, "stride": stride, "tower_type": tower_type, "state_exact": exact}
                )
                if not exact:
                    raise RuntimeError(f"common head mismatch c3={c3} stride={stride} {tower_type}")
    semantic = verify_semantic_body_initialization(cells)
    return {"fixed_p2_pairs": rows, "common_stride_heads": common_head_rows, "semantic_body": semantic}


def raw_forward_qc(cell: BuiltCell, image_size: int = 768, device: str = "cpu") -> dict[str, Any]:
    model = cell.model.to(device)
    training_flags = {name: bool(module.training) for name, module in model.named_modules()}
    model.eval()
    detect = model.model[-1]
    x = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32, device=device)
    old_training = bool(detect.training)
    try:
        detect.training = True
        with torch.inference_mode():
            raw = model(x)
            combined = [torch.cat((detect.cv2[i](feat), detect.cv3[i](feat)), 1) for i, feat in enumerate(raw["feats"])]
    finally:
        detect.training = old_training
        for name, module in model.named_modules():
            module.training = training_flags[name]
    if not isinstance(raw, dict) or set(raw) != {"boxes", "scores", "feats"}:
        raise RuntimeError(f"{cell.spec.cell_id}: unexpected raw output")
    strides = [int(float(v)) for v in detect.stride]
    spatial = [[image_size // stride, image_size // stride] for stride in strides]
    anchors = sum(height * width for height, width in spatial)
    expected_combined = [[1, NC + 4 * REG_MAX, height, width] for height, width in spatial]
    observed_combined = [list(value.shape) for value in combined]
    finite = bool(torch.isfinite(raw["boxes"]).all() and torch.isfinite(raw["scores"]).all())
    finite = finite and all(bool(torch.isfinite(value).all()) for value in combined)
    exact = (
        list(raw["boxes"].shape) == [1, 4 * REG_MAX, anchors]
        and list(raw["scores"].shape) == [1, NC, anchors]
        and observed_combined == expected_combined
    )
    if not finite or not exact:
        raise RuntimeError(f"{cell.spec.cell_id}: raw forward QC failed")
    report = {
        "cell_id": cell.spec.cell_id,
        "device": device,
        "image_size": image_size,
        "strides": strides,
        "boxes_shape": list(raw["boxes"].shape),
        "scores_shape": list(raw["scores"].shape),
        "combined_per_level": observed_combined,
        "finite": finite,
        "exact": exact,
        "module_training_flags_restored": True,
    }
    cell.model = model.cpu()
    del raw, combined, x
    return report


def release_cells(cells: dict[str, BuiltCell]) -> None:
    for cell in cells.values():
        cell.model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
