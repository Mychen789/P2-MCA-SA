"""Per-class AP comparison across experiments.

Reads metrics/AP50-95_class{0,1,2}(B) columns from each
runs/E0X_*_eval_test/results.csv and produces:
  - runs/_summary/per_class_ap.csv
  - runs/_summary/per_class_ap.png   grouped bar chart, x=experiment, hue=class
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nmv.utils.metrics_io import read_best

CLASSES = ["ebike", "bicycle", "etrike"]
_EVAL_SUFFIX = re.compile(r"_eval_(test|val)(_\w+)?$")


def imgsz_of(run_dir):
    """权威 imgsz：训练 run 的 args.yaml；回退 `_1280` 名称启发。审计 H4。"""
    train = run_dir.parent / _EVAL_SUFFIX.sub("", run_dir.name)
    args = train / "args.yaml"
    if args.exists():
        m = re.search(r"^imgsz:\s*(\d+)", args.read_text(encoding="utf-8", errors="ignore"), re.M)
        if m:
            return int(m.group(1))
    return 1280 if "_1280" in run_dir.name else 960


def plot_group(rows, out_dir, imgsz):
    names = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows])
    n_exp, n_cls = vals.shape
    x = np.arange(n_exp)
    width = 0.8 / n_cls
    fig, ax = plt.subplots(figsize=(max(10, n_exp * 1.2), 5))
    colors = ["#3a7bd5", "#48b56a", "#e6783f"]
    for cid in range(n_cls):
        ax.bar(x + (cid - n_cls / 2 + 0.5) * width, vals[:, cid],
               width=width, label=CLASSES[cid], color=colors[cid % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AP@0.5:0.95 (test split)")
    ax.set_title(f"Per-class AP across experiments (imgsz={imgsz})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = out_dir / f"per_class_ap_{imgsz}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_png}")


def main():
    runs = ROOT / "runs"
    out_dir = runs / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in sorted(runs.glob("E*_eval_test")):
        if d.name.endswith("_softnms") or d.name.endswith("_tta"):
            continue
        m = read_best(d / "results.csv")
        if m is None:
            continue
        per = [float(m.get(f"metrics/AP50-95_class{cid}(B)", 0.0)) for cid in range(len(CLASSES))]
        rows.append((d.name.replace("_eval_test", ""), imgsz_of(d), per))

    if not rows:
        print("No *_eval_test runs found.")
        return

    # csv 带 imgsz 列，按 (分辨率, 名称) 排序
    rows.sort(key=lambda r: (r[1], r[0]))
    csv = ["name,imgsz," + ",".join(f"AP_{c}" for c in CLASSES)]
    for name, imgsz, per in rows:
        csv.append(f"{name},{imgsz}," + ",".join(f"{v:.4f}" for v in per))
    (out_dir / "per_class_ap.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    print(f"Saved -> {out_dir / 'per_class_ap.csv'}")

    # 每个分辨率单独出图，杜绝 960/1280 混排
    for sz in sorted({r[1] for r in rows}):
        sub = [(n, per) for n, isz, per in rows if isz == sz]
        plot_group(sub, out_dir, sz)


if __name__ == "__main__":
    main()
