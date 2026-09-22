"""nmv package init: clamp GPU memory before patches load."""
import os as _os

import torch as _torch

# Clamp the PyTorch GPU allocator to a safe fraction of physical VRAM.
# On an 8 GB Windows GPU the ceiling PyTorch sees by default is 8 GB of physical VRAM
# plus roughly 8 GB of Windows "shared GPU memory" (system RAM mapped in), about 16 GB.
# Once an allocation spills out of the 8 GB of physical VRAM into that shared region the
# CUDA driver enters a sticky error state after some submission, and every later CUDA call
# - empty_cache and .cpu() copies included - returns OOM at once, aborting training.
#
# set_per_process_memory_fraction makes the allocator raise a catchable Python
# torch.cuda.OutOfMemoryError when the cap is exceeded, rather than a driver-level
# AcceleratorError, which is what lets the CPU fallback path in TAL work.
#
# Tunable through NMV_GPU_LIMIT_GB (default 7.0 GB, leaving 1 GB of headroom for the
# cuDNN workspace and the driver).
if _torch.cuda.is_available():
    _props = _torch.cuda.get_device_properties(0)
    _target_gb = float(_os.environ.get("NMV_GPU_LIMIT_GB", "7.0"))
    _fraction = max(0.1, min(0.98, (_target_gb * 1e9) / _props.total_memory))
    _torch.cuda.set_per_process_memory_fraction(_fraction)
    print(
        f"[nmv] GPU memory hard-limited to {_target_gb:.1f} GB "
        f"(fraction={_fraction:.3f}, device={_props.name}, "
        f"detected_total={_props.total_memory / 1e9:.2f}GB)"
    )

from nmv.patches import apply_all  # noqa: E402

apply_all()
