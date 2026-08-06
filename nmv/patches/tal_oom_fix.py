"""Patch ultralytics TaskAlignedAssigner OOM fallback to free GPU memory.

Why this exists:
  In ultralytics 8.4.x utils/tal.py:97-106, the GPU-OOM-then-CPU fallback
  catches OOM and re-runs on CPU, but does NOT call torch.cuda.empty_cache()
  before the CPU retry. The failed GPU tensors leave fragmented reserved
  blocks. The very NEXT iteration's allocator hits this fragmentation and
  fails on a tiny op (~5KB index tensor write at line 193).

  Observed on RTX 5060 Laptop 8GB: batch 12 OOM → CPU fallback succeeds →
  batch 13 dies on `ind[1] = gt_labels.squeeze(-1)`.

Fix:
  Wrap TaskAlignedAssigner.forward to call _safe_empty_cache+gc.collect BEFORE
  the CPU fallback (releases failed GPU tensors), AND once more after the
  .to(device) copy back so the next iteration starts clean.

  All empty_cache calls go through _safe_empty_cache() because once the CUDA
  context enters sticky-error state (which happens when allocations escape
  into Windows shared GPU memory), even empty_cache itself raises OOM. We
  swallow that failure rather than letting it cascade — the parent
  set_per_process_memory_fraction in nmv/__init__.py is the real fix; this
  is just defense in depth.
"""
import gc

import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.tal import TaskAlignedAssigner

_INSTALLED = False


def _safe_empty_cache():
    """empty_cache that survives CUDA-context corruption (sticky OOM error state)."""
    try:
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 — intentional: any failure is non-fatal here
        LOGGER.warning(f"[nmv] empty_cache failed (CUDA may be in sticky-error state): {e}")


def _safe_synchronize():
    try:
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"[nmv] cuda.synchronize failed: {e}")


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _orig_forward = TaskAlignedAssigner.forward

    def _patched_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        try:
            return _orig_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            LOGGER.warning("[nmv] TAL OOM caught — defragging then CPU retry")
            gc.collect()
            _safe_empty_cache()
            _safe_synchronize()
            device = gt_bboxes.device
            cpu_tensors = [
                t.detach().cpu()
                for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
            ]
            del pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
            gc.collect()
            _safe_empty_cache()
            result = _orig_forward(self, *cpu_tensors)
            out = tuple(t.to(device) for t in result)
            del cpu_tensors, result
            gc.collect()
            _safe_empty_cache()
            return out

    TaskAlignedAssigner.forward = _patched_forward
    _INSTALLED = True
    LOGGER.info("[nmv] TAL OOM-fallback defrag patch installed")
