"""Patch ultralytics for safe interrupt-resume cycle.

Two related bugs in ultralytics 8.4.x that break long training resume:

Bug 1: EarlyStopping state lost on resume.
  ultralytics creates a fresh EarlyStopping(best_fitness=0, best_epoch=0) every
  time BaseTrainer.__init__ runs, and resume_training() restores self.best_fitness
  from the checkpoint but does NOT restore the stopper's own state. Consequence:
  every resume zeros the patience counter, so early stopping never triggers on
  frequently interrupted runs.

Bug 2: final_eval() runs even on KeyboardInterrupt → strip_optimizer() called →
  last.pt loses optimizer/epoch/ema → resume permanently broken.
  ultralytics' BaseTrainer puts final_eval() in a finally block (or unconditionally
  after _do_train), so Ctrl+C during training causes final_eval to strip the
  optimizer state from last.pt. After that, model.train(resume=True) sees
  "not a resumable training checkpoint" and silently falls back to defaults
  (epochs=100, batch=16, coco8.yaml). Devastating for long experiments.

This patch:
  1. Wraps BaseTrainer.save_model to append stopper state into last.pt/best.pt
  2. Wraps BaseTrainer.resume_training to restore stopper state after resume
  3. Wraps BaseTrainer.final_eval to skip strip_optimizer when training was
     interrupted (epoch < epochs - 1 AND not naturally early-stopped)
"""
from pathlib import Path

import torch
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER

_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return

    _orig_save_model = BaseTrainer.save_model
    _orig_resume_training = BaseTrainer.resume_training
    _orig_final_eval = BaseTrainer.final_eval

    def _patched_save_model(self):
        _orig_save_model(self)
        stopper = getattr(self, "stopper", None)
        if stopper is None:
            return
        extra = {
            "stopper_best_fitness": float(stopper.best_fitness),
            "stopper_best_epoch": int(stopper.best_epoch),
        }
        targets = []
        last_path = getattr(self, "last", None)
        if last_path and Path(last_path).exists():
            targets.append(last_path)
        best_path = getattr(self, "best", None)
        if (
            self.best_fitness == self.fitness
            and best_path
            and Path(best_path).exists()
        ):
            targets.append(best_path)
        for pt_path in targets:
            try:
                ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
                ckpt.update(extra)
                torch.save(ckpt, pt_path)
            except Exception as e:
                LOGGER.warning(f"[nmv] Failed to persist stopper state into {pt_path}: {e}")

    def _patched_resume_training(self, ckpt):
        _orig_resume_training(self, ckpt)
        # post-resume defrag：load_state_dict 把 optimizer/scaler/ema 灌进 GPU
        # 会留下大量临时副本占着 reserved 块，立即清掉避免 first-batch 撞碎片化现场
        if torch.cuda.is_available():
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            LOGGER.info("[nmv] post-resume defrag complete")
        if not isinstance(ckpt, dict):
            return
        stopper = getattr(self, "stopper", None)
        if stopper is None or "stopper_best_fitness" not in ckpt:
            return
        stopper.best_fitness = float(ckpt["stopper_best_fitness"])
        stopper.best_epoch = int(ckpt.get("stopper_best_epoch", 0))
        LOGGER.info(
            f"[nmv] EarlyStopping state restored: "
            f"best_epoch={stopper.best_epoch}, "
            f"best_fitness={stopper.best_fitness:.4f}"
        )

    def _patched_final_eval(self):
        """Skip strip_optimizer if training was interrupted (preserve resumability).

        Detection: training is "interrupted" if current epoch hasn't reached the
        configured limit AND the EarlyStopping callback didn't legitimately fire.
        In that case we skip the entire final_eval (which contains the destructive
        strip_optimizer calls) and just save the current state via save_model.
        """
        epochs = getattr(self, "epochs", None)
        epoch = getattr(self, "epoch", None)
        stopper = getattr(self, "stopper", None)
        early_stopped = bool(getattr(stopper, "possible_stop", False)) if stopper else False
        if epochs is not None and epoch is not None:
            if (epoch < epochs - 1) and not early_stopped:
                LOGGER.warning(
                    f"[nmv] Training interrupted at epoch {epoch}/{epochs}: "
                    f"skipping final_eval to preserve resumable last.pt "
                    f"(strip_optimizer would otherwise nuke optimizer/epoch state)"
                )
                try:
                    self.save_model()
                except Exception as e:
                    LOGGER.warning(f"[nmv] save_model on interrupt also failed: {e}")
                return
        _orig_final_eval(self)

    BaseTrainer.save_model = _patched_save_model
    BaseTrainer.resume_training = _patched_resume_training
    BaseTrainer.final_eval = _patched_final_eval
    _INSTALLED = True
    LOGGER.info("[nmv] EarlyStopping + final_eval guard patches installed")
