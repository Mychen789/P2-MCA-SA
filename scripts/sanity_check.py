"""Verify the patch chain is working before any real training.

What this checks (in order):
  1. nmv import succeeds and triggers patch injection
  2. EMA / SimAM / MCA / ASPPLite are present in ultralytics.nn.tasks namespace
  3. parse_model has been re-execed with them in base_modules
  4. Each model yaml (baseline / +P2 / +P2+EMA / +P2+GFPN / +MCA / +MCA+GFPN / +MCA+CAGFPN) parses
  5. The +EMA / +MCA yamls instantiate EMA/MCA; +CAGFPN instantiates ASPPLite
  6. MPDIoU + Soft-NMS install/uninstall is reversible
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nmv  # triggers apply_all()


def step(label):
    print(f"\n=== {label} ===")


def main():
    step("1. Verify patch namespace injection")
    import ultralytics.nn.tasks as T
    for cls_name in ("EMA", "SimAM", "MCA", "ASPPLite", "P5Transformer"):
        assert hasattr(T, cls_name), f"{cls_name} not injected into ultralytics.nn.tasks"
        print(f"  {cls_name:14s}-> ", getattr(T, cls_name))

    step("2. Verify parse_model was replaced via exec (no source file)")
    fn_file = T.parse_model.__code__.co_filename
    assert fn_file == "<string>", (
        f"parse_model was not re-execed by register_modules (co_filename={fn_file})"
    )
    print(f"  parse_model.__code__.co_filename = {fn_file}  (re-execed in tasks namespace)")

    step("3. Parse each model yaml")
    from ultralytics import YOLO
    cfgs = [
        ("E01 baseline (yolov8m)",     "yolov8m.yaml",                        None),
        ("E02 +P2",                    "yolov8m-p2.yaml",                     None),
        ("E03 +P2+EMA",                "yolov8m-p2-ema.yaml",                 "EMA"),
        ("E05 +P2+EMA+GFPN",           "yolov8m-p2-ema-gfpn.yaml",            "EMA"),
        ("E08 +P2+MCA",                "yolov8m-p2-mca.yaml",                 "MCA"),
        ("E10 +P2+MCA+GFPN",           "yolov8m-p2-mca-gfpn.yaml",            "MCA"),
        ("E11 +P2+MCA+CAGFPN (Full)",  "yolov8m-p2-mca-cagfpn.yaml",          "ASPPLite"),
        ("E17 +P2+MCA+P5Transformer",  "yolov8m-p2-mca-p5trans.yaml",         "P5Transformer"),
        ("E24 v11m+P2+MCA",            "yolo11m-p2-mca.yaml",                  "MCA"),
    ]
    for label, cfg_name, required_cls in cfgs:
        cfg = cfg_name if cfg_name == "yolov8m.yaml" else str(ROOT / "configs" / "models" / cfg_name)
        m = YOLO(cfg)
        n_params = sum(p.numel() for p in m.model.parameters())
        print(f"  {label:30s} params={n_params/1e6:6.2f}M")
        if required_cls:
            count = sum(1 for mod in m.model.modules() if mod.__class__.__name__ == required_cls)
            assert count > 0, f"{label}: {required_cls} layer was NOT instantiated"
            print(f"  {'':30s} {required_cls} layers found: {count}")

    step("3b. P5Transformer forward pass smoke test")
    import torch
    p5t = T.P5Transformer(c1=768, c2=768, dim=192, n_layers=2, n_heads=4)
    p5t.eval()
    x = torch.randn(2, 768, 30, 30)
    with torch.no_grad():
        y = p5t(x)
    assert y.shape == x.shape, f"P5Transformer changed shape: {x.shape} -> {y.shape}"
    n_params = sum(p.numel() for p in p5t.parameters())
    print(f"  forward OK: in={tuple(x.shape)} out={tuple(y.shape)}  params={n_params/1e6:.3f}M")
    # At init proj_out is zeroed, so output must equal input exactly.
    assert torch.allclose(y, x), "P5Transformer at init should be identity (zero-init proj_out)"
    print(f"  init-as-identity OK: output exactly == input at init")

    step("4. Verify MPDIoU patch is reversible")
    from ultralytics.utils.loss import BboxLoss
    from nmv.patches.mpdiou_loss import install as install_mpdiou
    orig_name = BboxLoss.forward.__qualname__
    install_mpdiou(enable=True)
    assert BboxLoss.forward.__qualname__.endswith("_mpdiou_forward"), (
        f"MPDIoU patch failed: forward is {BboxLoss.forward.__qualname__}"
    )
    print(f"  MPDIoU enabled: BboxLoss.forward = {BboxLoss.forward.__qualname__}")
    install_mpdiou(enable=False)
    assert BboxLoss.forward.__qualname__ == orig_name, "MPDIoU uninstall did not restore original"
    print(f"  MPDIoU disabled: BboxLoss.forward = {BboxLoss.forward.__qualname__}")

    step("5. Verify Soft-NMS patch is reversible")
    import ultralytics.utils.nms as nms_mod
    from nmv.patches.soft_nms import install as install_soft
    orig_nms = nms_mod.non_max_suppression
    install_soft(enable=True)
    assert nms_mod.non_max_suppression.__name__ == "_soft_nms"
    print(f"  Soft-NMS enabled: non_max_suppression = {nms_mod.non_max_suppression.__name__}")
    install_soft(enable=False)
    assert nms_mod.non_max_suppression is orig_nms
    print(f"  Soft-NMS disabled: non_max_suppression = {nms_mod.non_max_suppression.__name__}")

    print("\n[OK] All sanity checks passed.")


if __name__ == "__main__":
    main()
