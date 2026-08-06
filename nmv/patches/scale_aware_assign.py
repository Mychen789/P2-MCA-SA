"""Scale-aware label assignment patch (size-range constrained TAL).

Diagnosed root cause (2026-05-22): adding the high-resolution P2 head drops
mAP_large ~20pp (E01 0.509 -> E02 0.303) BEFORE any attention module. The P2
anchors (stride 4) compete for / get assigned large objects they cannot regress
(reg_max*stride limit) and starve the low-resolution P5 head. Neither MPDIoU nor
GFPN repairs this once a strong attention (MCA) is present.

Fix: restrict label assignment so each detection level is only assigned objects
within its size range. We apply an UPPER bound only (lo=0 by default): a level of
stride s may be assigned a GT iff the GT's max side < HI_RATIO * s. This removes
only (large object, low-level) pairs and never deletes a level option for
small/medium objects, so small-object supervision is untouched.

Implementation notes:
  - The assigner.forward signature does NOT carry per-anchor stride, so we wrap
    make_anchors to stash the stride_tensor (same order as anc_points).
  - We wrap get_pos_mask and AND a size-range mask into mask_in_gts.

Size ranges are scale-relative (multiples of stride) -> imgsz-robust.
Env knobs:
  NMV_SCALE_ASSIGN=1        enable the scale-aware restriction
  NMV_SCALE_HI_RATIO=16.0   max GT side allowed at level s is HI_RATIO*s
  NMV_SCALE_LO_RATIO=0.0    min GT side allowed at level s is LO_RATIO*s
                            (largest stride always has hi=inf; smallest always lo=0)
  NMV_TAL_AUDIT=1           record pre-mask, post-mask and post-top-k assignment
                            counts by GT size and detection level
  NMV_TAL_AUDIT_PATH=...    JSON output path (set by the supplement runner)
  NMV_TAL_AUDIT_EVERY=100   checkpoint the JSON every N assignment calls
  NMV_TAL_AUDIT_TRAIN_ONLY=1 ignore assignment calls made by validation loss
  NMV_TAL_AUDIT_RESET=1     ignore an existing JSON instead of resume-merging it
"""
import atexit
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

import ultralytics.utils.loss as _loss_mod
import ultralytics.utils.tal as _tal_mod
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils import LOGGER
from ultralytics.utils.tal import TaskAlignedAssigner

_INSTALLED = False
_ORIG_GET_POS_MASK = None
_ORIG_MAKE_ANCHORS = None
_ORIG_MODEL_LOSS = None
_LAST = {"stride_tensor": None, "model_training": None}
_SCALE_ENABLED = False
_AUDIT_ENABLED = False
_AUDIT_ATEXIT_REGISTERED = False
_AUDIT = {
    "calls": 0,
    "fallback_gt": 0,
    "strides": [],
    "gt": [0.0, 0.0, 0.0],
    "candidate_pre": [],
    "candidate_post": [],
    "positive_post_topk": [],
    "alignment_sum": [],
    "overlap_sum": [],
}


def _zero_matrix(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _audit_output_path():
    return Path(os.environ.get(
        "NMV_TAL_AUDIT_PATH", f"runs/tal_assignment_audit_{os.getpid()}.json"
    ))


def _restore_existing_audit(strides):
    """Merge a previous process segment when long training resumes."""
    if os.environ.get("NMV_TAL_AUDIT_RESET", "0") == "1":
        return
    output = _audit_output_path()
    if not output.exists():
        return
    try:
        old = json.loads(output.read_text(encoding="utf-8"))
        settings = old.get("settings", {})
        expected_columns = [f"stride_{int(s)}" for s in strides]
        compatible = (
            old.get("schema_version") == 1
            and settings.get("matrix_columns") == expected_columns
            and bool(settings.get("scale_assign_enabled")) == _SCALE_ENABLED
            and abs(float(settings.get("hi_ratio", -1)) - float(os.environ.get("NMV_SCALE_HI_RATIO", "16.0"))) < 1e-9
            and abs(float(settings.get("lo_ratio", -1)) - float(os.environ.get("NMV_SCALE_LO_RATIO", "0.0"))) < 1e-9
        )
        if not compatible:
            LOGGER.warning(f"[nmv] TAL audit not resume-merged because settings changed: {output}")
            return

        size_names = ("small", "medium", "large")

        def read_table(section, key):
            table = section.get(key, {})
            return [[float(table.get(size, {}).get(col, 0.0)) for col in expected_columns]
                    for size in size_names]

        counts = old.get("counts", {})
        derived = old.get("derived", {})
        _AUDIT["calls"] = int(old.get("assignment_calls", 0))
        _AUDIT["fallback_gt"] = int(old.get("fallback_gt", 0))
        old_gt = old.get("gt_occurrences", {})
        _AUDIT["gt"] = [float(old_gt.get(size, 0)) for size in size_names]
        _AUDIT["candidate_pre"] = read_table(counts, "candidate_before_sa")
        _AUDIT["candidate_post"] = read_table(counts, "candidate_after_sa")
        _AUDIT["positive_post_topk"] = read_table(counts, "positive_after_topk")
        mean_alignment = read_table(derived, "mean_alignment_score_of_positives")
        mean_overlap = read_table(derived, "mean_iou_of_positives")
        for i in range(3):
            for j in range(len(strides)):
                positives = _AUDIT["positive_post_topk"][i][j]
                _AUDIT["alignment_sum"][i][j] = mean_alignment[i][j] * positives
                _AUDIT["overlap_sum"][i][j] = mean_overlap[i][j] * positives
        LOGGER.info(
            f"[nmv] resumed TAL audit totals from {output} "
            f"(assignment_calls={_AUDIT['calls']})"
        )
    except Exception as exc:
        LOGGER.warning(f"[nmv] TAL audit resume-merge failed; starting a new segment: {exc}")


def _ensure_audit_shape(strides):
    """Initialize audit matrices once the model's actual strides are known."""
    if not _AUDIT["strides"]:
        _AUDIT["strides"] = [float(s) for s in strides]
        for key in (
            "candidate_pre",
            "candidate_post",
            "positive_post_topk",
            "alignment_sum",
            "overlap_sum",
        ):
            _AUDIT[key] = _zero_matrix(3, len(strides))
        _restore_existing_audit(strides)
    return _AUDIT["strides"] == [float(s) for s in strides]


def _add_vector(dst, src):
    for i, value in enumerate(src):
        dst[i] += float(value)


def _add_matrix(dst, src):
    for i, row in enumerate(src):
        for j, value in enumerate(row):
            dst[i][j] += float(value)


def _audit_snapshot():
    size_names = ("small", "medium", "large")
    stride_names = [f"stride_{int(s)}" for s in _AUDIT["strides"]]

    def as_table(values):
        return {
            size_names[i]: {stride_names[j]: values[i][j] for j in range(len(stride_names))}
            for i in range(3)
        }

    positives = _AUDIT["positive_post_topk"]
    alignment = _AUDIT["alignment_sum"]
    overlap = _AUDIT["overlap_sum"]
    pos_per_gt = _zero_matrix(3, len(stride_names))
    mean_alignment = _zero_matrix(3, len(stride_names))
    mean_overlap = _zero_matrix(3, len(stride_names))
    level_fraction = _zero_matrix(3, len(stride_names))
    for i in range(3):
        row_total = sum(positives[i])
        for j in range(len(stride_names)):
            p = positives[i][j]
            pos_per_gt[i][j] = p / _AUDIT["gt"][i] if _AUDIT["gt"][i] else 0.0
            mean_alignment[i][j] = alignment[i][j] / p if p else 0.0
            mean_overlap[i][j] = overlap[i][j] / p if p else 0.0
            level_fraction[i][j] = p / row_total if row_total else 0.0

    return {
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "scale_assign_enabled": _SCALE_ENABLED,
            "hi_ratio": float(os.environ.get("NMV_SCALE_HI_RATIO", "16.0")),
            "lo_ratio": float(os.environ.get("NMV_SCALE_LO_RATIO", "0.0")),
            "size_definition": "COCO area bins in the current training-input pixel coordinates: small < 32^2, medium < 96^2, large >= 96^2",
            "training_grad_context_only": os.environ.get("NMV_TAL_AUDIT_TRAIN_ONLY", "1") == "1",
            "matrix_columns": stride_names,
        },
        "assignment_calls": int(_AUDIT["calls"]),
        "gt_occurrences": {size_names[i]: int(_AUDIT["gt"][i]) for i in range(3)},
        "fallback_gt": int(_AUDIT["fallback_gt"]),
        "counts": {
            "candidate_before_sa": as_table(_AUDIT["candidate_pre"]),
            "candidate_after_sa": as_table(_AUDIT["candidate_post"]),
            "positive_after_topk": as_table(positives),
        },
        "derived": {
            "positive_per_gt": as_table(pos_per_gt),
            "positive_level_fraction": as_table(level_fraction),
            "mean_alignment_score_of_positives": as_table(mean_alignment),
            "mean_iou_of_positives": as_table(mean_overlap),
        },
    }


def _write_audit():
    if not _AUDIT_ENABLED or not _AUDIT["calls"]:
        return
    output = _audit_output_path()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(json.dumps(_audit_snapshot(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(output)
    except Exception as exc:  # audit must never interrupt training
        LOGGER.warning(f"[nmv] TAL audit write failed: {exc}")


def flush_audit():
    """Public flush hook for short diagnostic jobs that do not rely on atexit."""
    _write_audit()


@torch.no_grad()
def _update_audit(mask_pre, mask_post, mask_pos, align_metric, overlaps, gt_bboxes, mask_gt, fallback_gt):
    """Accumulate assignment statistics with a single GPU-to-CPU synchronization."""
    st = _LAST["stride_tensor"]
    if st is None or st.shape[0] != mask_pos.shape[-1]:
        return
    stride_per_anchor = st.squeeze(-1).to(mask_pos.device)
    strides = sorted(float(s) for s in torch.unique(stride_per_anchor).detach().cpu().tolist())
    if not _ensure_audit_shape(strides):
        LOGGER.warning("[nmv] TAL audit skipped a batch because detection strides changed")
        return

    valid = mask_gt.squeeze(-1).bool()
    width = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp_min(0)
    height = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp_min(0)
    area = width * height
    size_hot = torch.stack((area < 32 ** 2, (area >= 32 ** 2) & (area < 96 ** 2), area >= 96 ** 2), dim=-1)
    size_hot = size_hot & valid.unsqueeze(-1)  # (batch, max_gt, 3)
    level_hot = torch.stack([stride_per_anchor == s for s in strides], dim=-1)  # (anchors, levels)
    size_f = size_hot.to(torch.float32)
    level_f = level_hot.to(torch.float32)

    def count_matrix(mask):
        return torch.einsum("bga,bgs,al->sl", mask.bool().to(torch.float32), size_f, level_f)

    pos_f = mask_pos.bool().to(torch.float32)
    gt_count = size_f.sum(dim=(0, 1))
    matrices = [
        count_matrix(mask_pre),
        count_matrix(mask_post),
        count_matrix(mask_pos),
        torch.einsum("bga,bgs,al->sl", align_metric * pos_f, size_f, level_f),
        torch.einsum("bga,bgs,al->sl", overlaps * pos_f, size_f, level_f),
    ]
    packed = torch.cat([gt_count.flatten()] + [m.flatten() for m in matrices]).detach().cpu().double().tolist()
    n_matrix = 3 * len(strides)
    _add_vector(_AUDIT["gt"], packed[:3])
    offset = 3
    for key in ("candidate_pre", "candidate_post", "positive_post_topk", "alignment_sum", "overlap_sum"):
        flat = packed[offset:offset + n_matrix]
        matrix = [flat[i * len(strides):(i + 1) * len(strides)] for i in range(3)]
        _add_matrix(_AUDIT[key], matrix)
        offset += n_matrix
    _AUDIT["calls"] += 1
    _AUDIT["fallback_gt"] += int(fallback_gt)

    every = max(1, int(os.environ.get("NMV_TAL_AUDIT_EVERY", "100")))
    if _AUDIT["calls"] % every == 0:
        _write_audit()


def _make_anchors_capture(feats, strides, grid_cell_offset=0.5):
    ap, st = _ORIG_MAKE_ANCHORS(feats, strides, grid_cell_offset)
    _LAST["stride_tensor"] = st  # (N, 1), order matches anc_points
    return ap, st


def _model_loss_capture(self, batch, preds=None):
    """Expose train/eval phase while BaseModel.loss calls the TAL assigner."""
    previous = _LAST["model_training"]
    _LAST["model_training"] = bool(self.training)
    try:
        return _ORIG_MODEL_LOSS(self, batch, preds)
    finally:
        _LAST["model_training"] = previous


def _scale_mask(self, gt_bboxes, n_anchors, device):
    """(b, n_max_boxes, n_anchors) bool mask of size-range-allowed GT/anchor pairs.

    Returns None if stride info is unavailable/mismatched (caller then falls back
    to unrestricted assignment for safety).
    """
    st = _LAST["stride_tensor"]
    if st is None or st.shape[0] != n_anchors:
        return None
    stride_per_anchor = st.squeeze(-1).to(device)  # (n_anchors,)

    lo_ratio = float(os.environ.get("NMV_SCALE_LO_RATIO", "0.0"))
    hi_ratio = float(os.environ.get("NMV_SCALE_HI_RATIO", "16.0"))

    strides = sorted({float(s) for s in self.stride})
    smallest, largest = strides[0], strides[-1]

    lo = torch.zeros_like(stride_per_anchor)
    hi = torch.full_like(stride_per_anchor, float("inf"))
    for s in strides:
        sel = stride_per_anchor == s
        lo[sel] = 0.0 if s == smallest else lo_ratio * s
        hi[sel] = float("inf") if s == largest else hi_ratio * s

    gw = gt_bboxes[..., 2] - gt_bboxes[..., 0]
    gh = gt_bboxes[..., 3] - gt_bboxes[..., 1]
    gsize = torch.maximum(gw, gh).unsqueeze(-1)  # (b, n_max_boxes, 1)

    return (gsize >= lo) & (gsize < hi)  # (b, n_max_boxes, n_anchors)


def _patched_get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
    mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
    mask_before_scale = mask_in_gts
    fallback_gt = 0

    # Scale-aware restriction (axis-aligned 4-coord boxes only; skip rotated).
    if _SCALE_ENABLED and gt_bboxes.shape[-1] == 4:
        allowed = _scale_mask(self, gt_bboxes, anc_points.shape[0], gt_bboxes.device)
        if allowed is not None:
            restricted = mask_in_gts * allowed.to(mask_in_gts.dtype)
            # Safety: if the restriction empties a GT that previously had candidate
            # anchors, keep the unrestricted mask for that GT (never lose a GT).
            had = mask_in_gts.sum(-1, keepdim=True) > 0
            now = restricted.sum(-1, keepdim=True) > 0
            keep_orig = had & (~now)
            fallback_gt = int(keep_orig.sum().item())
            mask_in_gts = torch.where(keep_orig, mask_in_gts, restricted)

    align_metric, overlaps = self.get_box_metrics(
        pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
    )
    mask_topk = self.select_topk_candidates(
        align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool()
    )
    mask_pos = mask_topk * mask_in_gts * mask_gt
    train_only = os.environ.get("NMV_TAL_AUDIT_TRAIN_ONLY", "1") == "1"
    if (_AUDIT_ENABLED and gt_bboxes.shape[-1] == 4
            and (not train_only or _LAST["model_training"] is not False)):
        _update_audit(
            mask_before_scale,
            mask_in_gts,
            mask_pos,
            align_metric,
            overlaps,
            gt_bboxes,
            mask_gt,
            fallback_gt,
        )
    return mask_pos, align_metric, overlaps


def install(enable=True, audit=False):
    """Install scale-aware assignment and/or the TAL assignment audit."""
    global _INSTALLED, _ORIG_GET_POS_MASK, _ORIG_MAKE_ANCHORS, _ORIG_MODEL_LOSS
    global _SCALE_ENABLED, _AUDIT_ENABLED, _AUDIT_ATEXIT_REGISTERED

    _SCALE_ENABLED = bool(enable)
    _AUDIT_ENABLED = bool(audit)

    if not enable and not audit:
        if _ORIG_GET_POS_MASK is not None:
            TaskAlignedAssigner.get_pos_mask = _ORIG_GET_POS_MASK
            _ORIG_GET_POS_MASK = None
        if _ORIG_MAKE_ANCHORS is not None:
            _tal_mod.make_anchors = _ORIG_MAKE_ANCHORS
            _loss_mod.make_anchors = _ORIG_MAKE_ANCHORS
            _ORIG_MAKE_ANCHORS = None
        if _ORIG_MODEL_LOSS is not None:
            BaseModel.loss = _ORIG_MODEL_LOSS
            _ORIG_MODEL_LOSS = None
        _INSTALLED = False
        return

    if _INSTALLED:
        return
    _ORIG_MAKE_ANCHORS = _tal_mod.make_anchors
    _tal_mod.make_anchors = _make_anchors_capture
    _loss_mod.make_anchors = _make_anchors_capture
    _ORIG_GET_POS_MASK = TaskAlignedAssigner.get_pos_mask
    TaskAlignedAssigner.get_pos_mask = _patched_get_pos_mask
    if _AUDIT_ENABLED:
        _ORIG_MODEL_LOSS = BaseModel.loss
        BaseModel.loss = _model_loss_capture
    _INSTALLED = True
    if _AUDIT_ENABLED and not _AUDIT_ATEXIT_REGISTERED:
        atexit.register(_write_audit)
        _AUDIT_ATEXIT_REGISTERED = True
    LOGGER.info(
        f"[nmv] TAL patch installed (scale_assign={_SCALE_ENABLED}, audit={_AUDIT_ENABLED}, "
        f"lo_ratio={os.environ.get('NMV_SCALE_LO_RATIO', '0.0')}, "
        f"hi_ratio={os.environ.get('NMV_SCALE_HI_RATIO', '16.0')})"
    )
