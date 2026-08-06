"""Stand-alone evaluation on the test split for any trained run.

Usage:
  python scripts/val.py --run E04_p2_ema_mpdiou_gfpn
  python scripts/val.py --run E04_p2_ema_mpdiou_gfpn --split test --tta
  python scripts/val.py --run E04_p2_ema_mpdiou_gfpn --soft-nms
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Run dir name under runs/, e.g. E04_p2_ema_mpdiou_gfpn")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--data", default=str(ROOT / "configs" / "data" / "nmv_visdrone_3cls.yaml"),
                   help="Dataset YAML. Defaults to NMV-SOD-3cls; use configs/data/visdrone10.yaml for official VisDrone val.")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="必须与训练分辨率一致!1280 训练的模型若用 960 评估,train/eval 分辨率不匹配会让整张表作废。")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (augment=True)")
    p.add_argument("--soft-nms", action="store_true", help="Enable Soft-NMS via env var")
    args = p.parse_args()

    if args.soft_nms:
        os.environ["NMV_NMS"] = "soft"
    import nmv  # apply patches
    from nmv.utils.data import ensure_data_yaml

    from ultralytics import YOLO

    data_yaml = ensure_data_yaml(Path(args.data))

    bp = ROOT / "runs" / args.run / "weights" / "best.pt"
    if not bp.exists():
        print(f"[ERROR] best.pt not found: {bp}")
        sys.exit(1)

    suffix_parts = [args.split]
    if args.tta:
        suffix_parts.append("tta")
    if args.soft_nms:
        suffix_parts.append("softnms")
    out_name = f"{args.run}_eval_{'_'.join(suffix_parts)}"

    model = YOLO(str(bp))
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        project=str(ROOT / "runs"),
        name=out_name,
        exist_ok=True,
        verbose=True,
        augment=args.tta,
    )

    # Persist a single-row results.csv so export_table.py picks up test metrics.
    # Header is kept compatible with ultralytics training results.csv so the
    # existing read_best() / export_table.py logic works unchanged.
    out_run = ROOT / "runs" / out_name
    try:
        b = metrics.box
        per_class = {int(c): float(ap) for c, ap in zip(b.ap_class_index, b.maps[b.ap_class_index])} \
            if hasattr(b, "maps") and len(b.ap_class_index) else {}
        header = ["epoch", "metrics/precision(B)", "metrics/recall(B)",
                  "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
        row = ["0", f"{float(b.mp):.6f}", f"{float(b.mr):.6f}",
               f"{float(b.map50):.6f}", f"{float(b.map):.6f}"]
        for cls_id in sorted(per_class):
            header.append(f"metrics/AP50-95_class{cls_id}(B)")
            row.append(f"{per_class[cls_id]:.6f}")
        (out_run / "results.csv").write_text(
            ",".join(header) + "\n" + ",".join(row) + "\n", encoding="utf-8"
        )
        print(f"[OK] Wrote results.csv (split={args.split})")
    except Exception as e:
        print(f"[WARN] could not write results.csv: {e}")

    print(f"\n[OK] Saved -> runs/{out_name}/")


if __name__ == "__main__":
    main()
