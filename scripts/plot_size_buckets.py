"""Grouped bar chart of COCO size-bucket mAP across experiments.

Reads:
  runs/E*_eval_test_buckets/metrics.json

Outputs:
  runs/_summary/size_buckets_bars.png        (论文主图: 11 个实验 × small/medium/large)
  runs/_summary/size_buckets_bars.pdf        (矢量版本, 便于 LaTeX)
  runs/_summary/size_buckets.csv             (柱状图底层数据, 论文表格可直接用)

故事线:
  - E01 (无 P2) large=0.509, small=0.200  -> small 弱、large 高
  - E02 (+P2)  large 砸到 0.303,  small 涨到 0.215
  - 后续 E03-E11 在 P2 基础上加各种模块, large 持续在 0.25-0.30 徘徊
  - E16_hi32 把 large 救回 0.39 (但仍未追平 baseline 0.509)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "_summary"

# 论文展示顺序：按"baseline → P2 引入 → EMA 支系 → MCA 支系 → CAGFPN/隔离 → scale-aware"组织
EXP_ORDER = [
    ("E01_baseline",                "E01 baseline"),
    ("E02_p2",                      "E02 +P2"),
    ("E03_p2_ema",                  "E03 +EMA"),
    ("E04_p2_ema_mpdiou",           "E04 +MPDIoU"),
    ("E05_p2_ema_mpdiou_gfpn",      "E05 +GFPN"),
    ("E08_p2_mca",                  "E08 +MCA"),
    ("E09_p2_mca_mpdiou",           "E09 +MCA+MPDIoU"),
    ("E10_p2_mca_mpdiou_gfpn",      "E10 +MCA+GFPN"),
    ("E11_full_mca_cagfpn",         "E11 +CAGFPN"),
    ("E14_full_mca_cagfpn_isolated","E14 isolated"),
    ("E16_mca_scale_assign",        "E16 scale-asg (hi=16)"),
    ("E16_mca_scale_assign_hi32",   "E16' scale-asg (hi=32)"),
]

HIGHLIGHT = {"E08_p2_mca", "E16_mca_scale_assign_hi32"}


def load_metrics():
    out = []
    for key, label in EXP_ORDER:
        p = RUNS / f"{key}_eval_test_buckets" / "metrics.json"
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        m = json.loads(p.read_text())
        out.append((key, label, m["mAP_small"], m["mAP_medium"], m["mAP_large"], m["mAP_50"]))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_metrics()
    n = len(rows)
    if n == 0:
        print("[fatal] no metrics.json found"); return

    labels = [r[1] for r in rows]
    small  = np.array([r[2] for r in rows])
    medium = np.array([r[3] for r in rows])
    large  = np.array([r[4] for r in rows])

    x = np.arange(n)
    w = 0.27

    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=160)
    # 配色用 colorblind-friendly 三色 (cubehelix-like)
    c_small, c_med, c_large = "#5470c6", "#fac858", "#ee6666"
    bars_s = ax.bar(x - w, small,  w, label="small (area<32²)",  color=c_small)
    bars_m = ax.bar(x,     medium, w, label="medium (32²≤area<96²)", color=c_med)
    bars_l = ax.bar(x + w, large,  w, label="large (area≥96²)",  color=c_large)

    # 标注数值
    for bars in (bars_s, bars_m, bars_l):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.008,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="#333")

    # baseline E01 的 large 作为参考线，强调"大目标基线"
    e01_large = small[0]  # default, override below
    for r in rows:
        if r[0] == "E01_baseline":
            e01_large = r[4]; break
    ax.axhline(e01_large, ls="--", lw=0.9, color="#ee6666", alpha=0.55,
               label=f"E01 large baseline = {e01_large:.3f}")

    # 高亮主模型 x 标签
    for i, r in enumerate(rows):
        if r[0] in HIGHLIGHT:
            for bars in (bars_s, bars_m, bars_l):
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(1.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_ylim(0, max(large.max(), medium.max(), small.max()) * 1.18)
    ax.set_title("COCO size-bucket mAP across NMV-SOD ablation experiments (test split)", fontsize=11)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2)

    plt.tight_layout()
    png = OUT / "size_buckets_bars.png"
    pdf = OUT / "size_buckets_bars.pdf"
    plt.savefig(png, bbox_inches="tight"); plt.savefig(pdf, bbox_inches="tight")
    print(f"[OK] {png}\n[OK] {pdf}")

    # 输出底层 CSV (论文表格直接可用)
    csv = ["name,label,mAP_small,mAP_medium,mAP_large,mAP_50"]
    for key, label, s, m, l, m50 in rows:
        csv.append(f"{key},{label},{s:.4f},{m:.4f},{l:.4f},{m50:.4f}")
    (OUT / "size_buckets.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    print(f"[OK] {OUT/'size_buckets.csv'}")


if __name__ == "__main__":
    main()
