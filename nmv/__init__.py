"""nmv package init: clamp GPU memory before patches load."""
import os as _os

import torch as _torch

# 限制 PyTorch GPU 分配上限到物理 VRAM 的安全比例。
# 在 Windows 8GB GPU 上，PyTorch 默认看到的"上限"是 8GB 物理 + ~8GB Windows
# "shared GPU memory" (系统 RAM 映射) ≈ 16GB。一旦分配溢出 8GB 物理 VRAM 进入
# shared 区域，CUDA driver 在某次提交后进入 sticky error state，所有后续 CUDA
# 调用（包括 empty_cache、.cpu() 拷贝等无害操作）都会立即返回 OOM error，
# 训练进程必须 abort。
#
# set_per_process_memory_fraction 让 PyTorch allocator 在超限时抛 catchable
# 的 Python torch.cuda.OutOfMemoryError 异常（而非 driver-level
# AcceleratorError），TAL 的 CPU 回退路径才能正常工作。
#
# 通过 NMV_GPU_LIMIT_GB 环境变量调节（默认 7.0 GB，留 1 GB buffer 给 cuDNN
# workspace + 驱动）。
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
