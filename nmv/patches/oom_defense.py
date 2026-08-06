"""Aggressive multi-hook defragmentation for 8GB GPU YOLOv8 training.

Why this exists:
  v1 fired empty_cache every 50 batches (on_train_batch_end). Crash observed
  at batch 13 — defrag never got a chance. Resume-path and OOM-fallback
  cascades need defrag at moments BEFORE the crash window.

Hook timing (priority order):
  1. on_train_start: defrag once before batch 0 (clears post-resume bloat)
  2. on_train_epoch_start: defrag at each epoch boundary
  3. on_train_batch_end: defrag every N batches (default 10, was 50)

Each defrag also runs gc.collect() to release Python-held CUDA tensor refs.
"""
import gc
import os

import torch
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER

_INSTALLED = False
_EVERY_N = int(os.environ.get("NMV_DEFRAG_EVERY", "10"))


def _defrag(tag=""):
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    if tag:
        LOGGER.info(f"[nmv] defrag@{tag}")


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _orig_init = BaseTrainer.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._oom_defense_step = 0

        def _on_train_start(trainer):
            _defrag("train_start")

        def _on_train_epoch_start(trainer):
            _defrag(f"epoch_{getattr(trainer, 'epoch', '?')}_start")

        def _on_train_batch_end(trainer):
            trainer._oom_defense_step += 1
            if trainer._oom_defense_step % _EVERY_N == 0:
                _defrag()

        self.add_callback("on_train_start", _on_train_start)
        self.add_callback("on_train_epoch_start", _on_train_epoch_start)
        self.add_callback("on_train_batch_end", _on_train_batch_end)

    BaseTrainer.__init__ = _patched_init
    _INSTALLED = True
    LOGGER.info(
        f"[nmv] OOM defense installed (every {_EVERY_N} batches + train/epoch start)"
    )
