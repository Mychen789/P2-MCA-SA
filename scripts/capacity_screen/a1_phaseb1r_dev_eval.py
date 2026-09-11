from __future__ import annotations

"""Blinded dev evaluation for the A1 capacity 2x2 factorial screen.

This is the evaluation adapter that ``b1r_runner.py`` reports as not
connected.  It runs only after all four cells are COMPLETE, and it touches
nothing the frozen design forbids:

* the evaluated weights are the four ``C**_FINAL_EPOCH_EMA_EPOCH_149.pt``
  files, never ``best.pt`` or ``last.pt`` -- the design sets
  ``best_by_dev_selection: false`` and
  ``recovery_checkpoint_used_for_model_selection: false``;
* the only images read are the 1322 DEV rows of
  ``phaseb1_dev_manifest.tsv``, verified by byte count and SHA-256; the
  official VisDrone val and test splits are never opened;
* the size buckets reproduce the frozen manifest definition exactly --
  box area after scaling by ``768 / max(width, height)``, split at
  ``32**2`` and ``96**2`` -- so ``AP_S_768`` here means the same thing as
  ``size_counts_768`` in the data freeze;
* detection settings match the project canonical bucket evaluator
  (``scripts/eval_size_buckets.py``): ``imgsz=768``, ``conf=0.001``,
  ``iou=0.7``, COCO AP through pycocotools.

Cell ``status.json`` files are left untouched: ``b1r_runner`` rejects any
status whose ``dev_metric_computed`` is not ``False``, so the result of
this pass lives in its own artefact.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

RUNTIME_ROOT = Path(__file__).resolve().parent
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

import a1_phaseb1r_cell as cell_policy
import a1_phaseb1r_contract as contract
import a1_phaseb1r_engine as engine
import a1_phaseb1r_numeric as numeric

SCHEMA = "A1_PHASEB1R_DEV_EVALUATION_V1"
DEV_ROWS = 1322
IMGSZ = 768
CONF = 0.001
NMS_IOU = 0.7
MAX_DET = 300
SMALL_AREA = 32 ** 2
MEDIUM_AREA = 96 ** 2
RESULT_ROOT = (
    numeric.PROJECT_ROOT / "runs" / "_root_cause_audit" / "B1_CAP_SCREEN_SIMPLE_20260901"
)
OUTPUT_NAME = "DEV_EVALUATION.json"
CHECKPOINT_SCHEMA = "A1_PHASEB1R_FINAL_EMA_CHECKPOINT_V1"
CHECKPOINT_KEYS = {
    "schema",
    "cell_id",
    "epoch_index",
    "ema_updates",
    "ema_state_sha256",
    "ema_state_dict",
}
CONTRAST_METRICS = ("AP_768", "AP50_768", "AP_S_768", "AP_M_768", "AP_L_768")


class DevEvalError(RuntimeError):
    """Raised when the evaluation boundary or a checkpoint identity differs."""


def require_four_complete(result_root: Path) -> None:
    for cell_id in contract.CELL_ORDER:
        status_path = result_root / cell_id / "status.json"
        if not status_path.is_file() or status_path.is_symlink():
            raise DevEvalError(f"missing status.json for {cell_id}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "COMPLETE" or status.get("completed_epochs") != 150:
            raise DevEvalError(f"{cell_id} is not a completed 150-epoch cell")
        if status.get("checkpoint_strict_reload_passed") is not True:
            raise DevEvalError(f"{cell_id} never passed strict checkpoint reload")
        if status.get("final_checkpoint") != contract.expected_checkpoint_name(cell_id):
            raise DevEvalError(f"{cell_id} final checkpoint name is not the frozen one")


def read_dev_images(verify_files: bool = True) -> tuple[Path, ...]:
    return engine.read_manifest_image_paths(
        engine.DEV_MANIFEST,
        data_root=engine.DATA_ROOT,
        expected_role="DEV",
        expected_rows=DEV_ROWS,
        verify_files=verify_files,
    )


def build_coco_ground_truth(images: tuple[Path, ...]) -> tuple[dict[str, Any], dict[str, int]]:
    """COCO ground truth with every box scaled into the frozen 768 frame."""
    from PIL import Image

    coco: dict[str, Any] = {
        "info": {"description": "A1 capacity screen dev split, boxes scaled to 768"},
        "images": [],
        "annotations": [],
        "categories": [{"id": index + 1, "name": str(index)} for index in range(10)],
    }
    bucket_counts = {"small": 0, "medium": 0, "large": 0}
    annotation_id = 1
    for image_id, image_path in enumerate(images, start=1):
        with Image.open(image_path) as handle:
            width, height = handle.size
        ratio = IMGSZ / max(width, height)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": int(round(width * ratio)),
                "height": int(round(height * ratio)),
            }
        )
        label_path = engine.DATA_ROOT / "labels" / "train" / f"{image_path.stem}.txt"
        seen: set[tuple[float, ...]] = set()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = tuple(float(value) for value in line.split())
            if len(parts) != 5:
                raise DevEvalError(f"malformed label row in {label_path}")
            if parts in seen:
                continue  # the freeze counts effective, de-duplicated boxes
            seen.add(parts)
            cls, x, y, w, h = parts
            box_w = w * width * ratio
            box_h = h * height * ratio
            x1 = (x * width - w * width / 2) * ratio
            y1 = (y * height - h * height / 2) * ratio
            area = box_w * box_h
            bucket = "small" if area < SMALL_AREA else "medium" if area < MEDIUM_AREA else "large"
            bucket_counts[bucket] += 1
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(cls) + 1,
                    "bbox": [float(x1), float(y1), float(box_w), float(box_h)],
                    "area": float(area),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return coco, bucket_counts


def load_cell_model(stack: Mapping[str, Any], cell_id: str, result_root: Path) -> tuple[Any, dict[str, Any]]:
    torch = stack["torch"]
    model, fresh_qc = numeric.build_fresh_cell(stack, cell_id)
    checkpoint_path = result_root / cell_id / contract.expected_checkpoint_name(cell_id)
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise DevEvalError(f"missing final checkpoint: {checkpoint_path}")
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if set(loaded) != CHECKPOINT_KEYS:
        raise DevEvalError(f"{cell_id} checkpoint keys are not exact")
    if loaded["schema"] != CHECKPOINT_SCHEMA or loaded["cell_id"] != cell_id:
        raise DevEvalError(f"{cell_id} checkpoint identity mismatch")
    if loaded["epoch_index"] != 149:
        raise DevEvalError(f"{cell_id} checkpoint is not the epoch-149 EMA")
    state = loaded["ema_state_dict"]
    digest = cell_policy.state_mapping_sha256(torch, state)
    if digest != loaded["ema_state_sha256"]:
        raise DevEvalError(f"{cell_id} EMA state hash mismatch")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise DevEvalError(f"{cell_id} strict EMA reload failed")
    qc = {
        "params_total": fresh_qc["params_total"],
        "strides": fresh_qc["strides"],
        "initial_state_sha256": fresh_qc["initial_state_sha256"],
        "ema_state_sha256": loaded["ema_state_sha256"],
        "ema_updates": int(loaded["ema_updates"]),
        "strict_reload": True,
        "checkpoint": checkpoint_path.name,
    }
    return model, qc


def predict_dev(
    stack: Mapping[str, Any],
    model: Any,
    images: tuple[Path, ...],
    coco: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from ultralytics import YOLO
    from ultralytics.utils import YAML

    torch = stack["torch"]
    names = YAML.load(numeric.DATA_YAML).get("names")
    if isinstance(names, list):
        names = {index: value for index, value in enumerate(names)}
    model.names = dict(names)
    model.eval()

    id_by_name = {row["file_name"]: row["id"] for row in coco["images"]}
    predictions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as temporary:
        weights = Path(temporary) / "cell_eval.pt"
        torch.save(
            {
                "model": model,
                "train_args": {"imgsz": IMGSZ, "data": str(numeric.DATA_YAML), "task": "detect"},
                "epoch": -1,
            },
            weights,
        )
        detector = YOLO(str(weights))
        batch = 8
        for start in range(0, len(images), batch):
            chunk = [str(path) for path in images[start : start + batch]]
            outputs = detector.predict(
                chunk,
                imgsz=IMGSZ,
                conf=CONF,
                iou=NMS_IOU,
                max_det=MAX_DET,
                verbose=False,
            )
            for path, output in zip(chunk, outputs):
                name = Path(path).name
                if output.boxes is None or len(output.boxes) == 0:
                    continue
                height, width = output.orig_shape
                ratio = IMGSZ / max(width, height)
                xyxy = output.boxes.xyxy.cpu().numpy()
                classes = output.boxes.cls.cpu().numpy()
                scores = output.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), cls, score in zip(xyxy, classes, scores):
                    predictions.append(
                        {
                            "image_id": id_by_name[name],
                            "category_id": int(cls) + 1,
                            "bbox": [
                                float(x1 * ratio),
                                float(y1 * ratio),
                                float((x2 - x1) * ratio),
                                float((y2 - y1) * ratio),
                            ],
                            "score": float(score),
                        }
                    )
            done = min(start + batch, len(images))
            if done % 160 == 0 or done == len(images):
                print(f"      {done}/{len(images)} images", flush=True)
    return predictions


def coco_metrics(gt_path: Path, predictions: list[dict[str, Any]], work_dir: Path) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    pred_path = work_dir / "predictions.json"
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))
    evaluation = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    stats = [float(value) for value in evaluation.stats]
    keys = (
        "AP_768",
        "AP50_768",
        "AP75_768",
        "AP_S_768",
        "AP_M_768",
        "AP_L_768",
        "AR_1_768",
        "AR_10_768",
        "AR_100_768",
        "AR_S_768",
        "AR_M_768",
        "AR_L_768",
    )
    metrics = {key: stats[index] for index, key in enumerate(keys)}
    metrics["predictions"] = len(predictions)
    return metrics


def factorial_contrasts(cells: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Cell ids are C<p2><wide>: C10 is P2 at c3=96, C01 is no P2 at c3=192."""

    def delta(metric: str, left: str, right: str) -> float:
        return round(cells[left][metric] - cells[right][metric], 6)

    return {
        "p2_effect_at_narrow": {m: delta(m, "C10", "C00") for m in CONTRAST_METRICS},
        "p2_effect_at_wide": {m: delta(m, "C11", "C01") for m in CONTRAST_METRICS},
        "width_effect_without_p2": {m: delta(m, "C01", "C00") for m in CONTRAST_METRICS},
        "width_effect_with_p2": {m: delta(m, "C11", "C10") for m in CONTRAST_METRICS},
        "p2_versus_width_matched": {m: delta(m, "C10", "C01") for m in CONTRAST_METRICS},
        "interaction": {
            m: round(
                (cells["C11"][m] - cells["C01"][m]) - (cells["C10"][m] - cells["C00"][m]),
                6,
            )
            for m in CONTRAST_METRICS
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Blinded dev evaluation of the A1 capacity 2x2 screen"
    )
    parser.add_argument("--result-root", default=str(RESULT_ROOT))
    parser.add_argument(
        "--skip-file-verification",
        action="store_true",
        help="skip per-image SHA-256 verification of the dev manifest",
    )
    parser.add_argument("--cells", default=",".join(contract.CELL_ORDER))
    args = parser.parse_args(argv)

    result_root = Path(args.result_root).resolve()
    requested = tuple(value.strip() for value in args.cells.split(",") if value.strip())
    if any(cell not in contract.CELL_ORDER for cell in requested):
        raise DevEvalError("unknown cell requested")

    print("[1/4] verifying the four training cells")
    require_four_complete(result_root)
    print("      four cells COMPLETE, strict reload passed, frozen checkpoint names present")

    print("[2/4] reading the frozen DEV manifest")
    images = read_dev_images(verify_files=not args.skip_file_verification)
    print(f"      {len(images)} dev images verified")

    print("[3/4] building COCO ground truth in the 768 reference frame")
    work_dir = result_root / "_dev_eval"
    work_dir.mkdir(exist_ok=True)
    coco, bucket_counts = build_coco_ground_truth(images)
    gt_path = work_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(coco), encoding="utf-8")
    print(
        f"      {len(coco['annotations'])} boxes  small/medium/large = "
        f"{bucket_counts['small']}/{bucket_counts['medium']}/{bucket_counts['large']}"
    )

    stack = numeric.lazy_import_numeric_stack()
    results: dict[str, Any] = {}
    qc_report: dict[str, Any] = {}
    for cell_id in requested:
        print(f"[4/4] {cell_id}: rebuilding the frozen architecture and loading the epoch-149 EMA")
        model, qc = load_cell_model(stack, cell_id, result_root)
        qc_report[cell_id] = qc
        print(
            f"      params={qc['params_total']:,} strides={qc['strides']} "
            f"ema_updates={qc['ema_updates']}"
        )
        predictions = predict_dev(stack, model, images, coco)
        cell_dir = work_dir / cell_id
        cell_dir.mkdir(exist_ok=True)
        results[cell_id] = coco_metrics(gt_path, predictions, cell_dir)
        print(
            f"      AP_S_768={results[cell_id]['AP_S_768']:.4f}  "
            f"AP_768={results[cell_id]['AP_768']:.4f}"
        )
        del model

    document = {
        "schema": SCHEMA,
        "experiment": "A1_CAPACITY_2X2_FACTORIAL_SCREEN",
        "result_root": str(result_root),
        "dev_manifest": str(engine.DEV_MANIFEST),
        "dev_images": len(images),
        "dev_boxes": len(coco["annotations"]),
        "dev_bucket_counts_768": bucket_counts,
        "settings": {
            "imgsz": IMGSZ,
            "conf": CONF,
            "nms_iou": NMS_IOU,
            "max_det": MAX_DET,
            "checkpoint_kind": "FINAL_EPOCH_EMA_EPOCH_149",
            "best_by_dev_selection": False,
            "official_val_accessed": False,
            "bucket_definition": "area after scaling by 768/max(width,height); 32^2 and 96^2 thresholds",
        },
        "primary_metric": "AP_S_768",
        "cell_specification": {
            cell: {"p2": cell in {"C10", "C11"}, "c3": 192 if cell in {"C01", "C11"} else 96}
            for cell in contract.CELL_ORDER
        },
        "checkpoint_qc": qc_report,
        "cells": results,
    }
    if set(results) == set(contract.CELL_ORDER):
        document["contrasts"] = factorial_contrasts(results)
    output = result_root / OUTPUT_NAME
    output.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
