"""Plot Params / GFLOPs vs test mAP scatter for size-vs-accuracy Pareto curve.

Reads:
  runs/_summary/complexity.csv          name, params_M, GFLOPs, FPS
  runs/_summary/comparison_test.csv     idx, name(*_eval_test), P, R, mAP50, mAP50_95

Outputs:
  runs/_summary/pareto_params.png
  runs/_summary/pareto_gflops.png
  runs/_summary/pareto_fps.png

E11 is highlighted so reviewers can see where it sits on the Pareto frontier
(needed when total mAP gain is small but efficiency/accuracy trade-off is the
real story).
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# 复杂度与速度的测量分辨率（bench_complexity.py 当前硬编码 960）。
# 精度散点必须取同一分辨率的 test 子表，否则会把 1280 行误标成 960（审计 H4）。
IMGSZ = 960

HIGHLIGHTS = {"E08_p2_mca", "E11_full_mca_cagfpn"}
EXTERNAL = {"E12_yolov8s_baseline", "E13_yolov8l_baseline"}


def load_complexity(p):
    out = {}
    with open(p, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["name"]] = dict(
                params=float(row["params_M"]),
                gflops=float(row["GFLOPs"]),
                fps=float(row["FPS"]),
            )
    return out


def load_test(p):
    out = {}
    with open(p, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["name"].replace("_eval_test", "")
            # skip variants (softnms, tta)
            if "_softnms" in name or "_tta" in name:
                continue
            out[name] = dict(map50=float(row["mAP50"]), map95=float(row["mAP50_95"]))
    return out


def scatter(ax, x, y, labels, x_label, y_label, title):
    for xi, yi, lab in zip(x, y, labels):
        if lab in HIGHLIGHTS:
            color, marker, size = "#d62728", "*", 220
        elif lab in EXTERNAL:
            color, marker, size = "#2ca02c", "s", 110
        else:
            color, marker, size = "#3a7bd5", "o", 90
        ax.scatter(xi, yi, c=color, marker=marker, s=size, edgecolor="white",
                   linewidth=1.2, zorder=3)
        short = lab.split("_", 1)[0]
        ax.annotate(short, (xi, yi), xytext=(6, 4), textcoords="offset points",
                    fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)


def main():
    summary = ROOT / "runs" / "_summary"
    cx = load_complexity(summary / "complexity.csv")
    # 取与 complexity 同分辨率的 test 子表（export_table.py 现按 imgsz 分表产出）；
    # 子表不存在时回退到合并表（旧行为，仅当全仓只有单一分辨率时安全）。
    test_csv = summary / f"comparison_test_{IMGSZ}.csv"
    if not test_csv.exists():
        test_csv = summary / "comparison_test.csv"
        print(f"[warn] {summary/('comparison_test_%d.csv'%IMGSZ)} 不存在，回退合并表（可能含其它分辨率）")
    tt = load_test(test_csv)
    names = [n for n in cx if n in tt]
    if not names:
        print(f"No overlap between complexity.csv and {test_csv.name}. Run "
              "scripts/bench_complexity.py and scripts/export_table.py first.")
        return

    params = [cx[n]["params"] for n in names]
    gflops = [cx[n]["gflops"] for n in names]
    fps = [cx[n]["fps"] for n in names]
    map95 = [tt[n]["map95"] for n in names]
    map50 = [tt[n]["map50"] for n in names]

    # 1) params vs mAP@0.5:0.95
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter(ax, params, map95, names,
            "Parameters (M)", f"test mAP@0.5:0.95 (imgsz={IMGSZ})",
            "Params vs accuracy (Pareto)")
    fig.tight_layout()
    fig.savefig(summary / "pareto_params.png", dpi=150)
    plt.close(fig)

    # 2) gflops vs mAP@0.5:0.95
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter(ax, gflops, map95, names,
            f"GFLOPs @ {IMGSZ}x{IMGSZ}", f"test mAP@0.5:0.95 (imgsz={IMGSZ})",
            "Compute vs accuracy (Pareto)")
    fig.tight_layout()
    fig.savefig(summary / "pareto_gflops.png", dpi=150)
    plt.close(fig)

    # 3) fps vs mAP@0.5
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter(ax, fps, map50, names,
            f"FPS @ batch=1 imgsz={IMGSZ} (RTX 5060)", f"test mAP@0.5 (imgsz={IMGSZ})",
            "Speed vs accuracy (Pareto)")
    fig.tight_layout()
    fig.savefig(summary / "pareto_fps.png", dpi=150)
    plt.close(fig)

    print("Saved 3 Pareto plots ->", summary)
    print("  pareto_params.png   params vs mAP@0.5:0.95")
    print("  pareto_gflops.png   GFLOPs vs mAP@0.5:0.95")
    print("  pareto_fps.png      FPS vs mAP@0.5")


if __name__ == "__main__":
    main()
