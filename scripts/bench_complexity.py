"""Benchmark Params, GFLOPs, and FPS for each completed experiment.

Uses ultralytics' built-in `model.info()` for params/FLOPs (which under the hood
uses thop), and a simple warmup-then-time loop for FPS at imgsz=960 batch=1.
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nmv
from ultralytics import YOLO


def measure_fps(model, imgsz=960, n_warmup=10, n_iter=50):
    p = next(model.model.parameters())
    x = torch.zeros(1, 3, imgsz, imgsz, device=p.device, dtype=p.dtype)
    with torch.no_grad():
        for _ in range(n_warmup):
            model.model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model.model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return n_iter / dt


def main():
    runs = ROOT / "runs"
    out_dir = runs / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in sorted(runs.glob("E*")):
        bp = d / "weights" / "best.pt"
        if not bp.exists():
            continue
        print(f"=== {d.name} ===")
        m = YOLO(str(bp))
        m.model.eval()
        info = m.model.info(imgsz=960, verbose=True)
        if info is None:
            n_params = sum(p.numel() for p in m.model.parameters())
            gflops = 0.0
        else:
            n_layers, n_params, n_grads, gflops = info
        if torch.cuda.is_available():
            m.model.cuda().half()
        try:
            fps = measure_fps(m, imgsz=960)
        except RuntimeError as ex:
            print("  fps measurement failed:", ex)
            fps = 0.0
        rows.append((d.name, n_params, gflops, fps))
        print(f"  params={n_params/1e6:.2f}M  GFLOPs={gflops:.2f}  FPS={fps:.1f}")

    csv_lines = ["name,params_M,GFLOPs,FPS"]
    for name, p, g, f in rows:
        csv_lines.append(f"{name},{p/1e6:.3f},{g:.3f},{f:.2f}")
    out_csv = out_dir / "complexity.csv"
    out_csv.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
