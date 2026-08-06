"""Aggregate multi-seed results into a mean ± std table + pairwise Welch t-tests.

**2026-07-12 rewrite — dual-backbone, COCO protocol.**
Source of truth is now the cocoapi size-bucket export
  runs/<exp>_eval_test_buckets/metrics.json            seed 42 (run name has no _sN)
  runs/<exp>_s<seed>_eval_test_buckets/metrics.json    seed 1 / 7 / ...
i.e. the SAME provenance as `results_tables_dualbackbone.md` (test split, imgsz=1280,
pycocotools). This keeps the aggregated numbers self-consistent with the paper tables
and — unlike the old ultralytics `_eval_test/results.csv` path — covers E24 and E25
(whose ultralytics results.csv is missing / not the canonical protocol).

Outputs:
  runs/_summary/seed_stats.csv          mean,std per (exp, metric)
  runs/_summary/seed_stats.md           paper-style markdown table
  runs/_summary/seed_significance.md    pairwise Welch t-tests on the two gated
                                        metrics (mAP@0.5:0.95 and mAP_small)

Pure JSON + math (no torch / GPU) — safe to run at any time, including while a
training run is using the GPU.
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Optional: use scipy for an exact Welch t-test / p-value when available;
# otherwise fall back to the crude normal approximation below.
try:
    from scipy import stats as _scipy_stats
except Exception:  # scipy not installed in this env
    _scipy_stats = None


# ── Experiments (dual-backbone main table) ────────────────────────────────────
# stem = run-dir prefix WITHOUT the _sN seed suffix and WITHOUT _eval_test_buckets.
# seed42 dir = <stem>_eval_test_buckets ; other seeds = <stem>_s<seed>_eval_test_buckets.
EXPS = [
    # stem,                                  display,                   backbone
    ("E01_baseline_1280",                    "E01 baseline",            "v8m"),   # 3 seed
    ("E02_p2_1280",                          "E02 +P2",                 "v8m"),   # 3 seed (seed7 last)
    ("E16_mca_scale_assign_hi32_1280",       "E16' hero (ours)",        "v8m"),   # 3 seed
    ("E17_p2_mca_p5trans_scale_1280",        "E17 +P5T (neg.)",         "v8m"),   # 1 seed (§6)
    ("E18_yolov11m_baseline_1280",           "E18 baseline (b2)",       "v11m"),  # 1 seed (ref)
    ("E18_yolov11m_baseline_b1_1280",        "E25 baseline (b1, fair)", "v11m"),  # 1 seed
    ("E24_yolo11m_p2_mca_scale_1280",        "E24 hero (ours)",         "v11m"),  # 3 seed
]

# cocoapi metrics.json key -> paper label
METRICS = [
    ("mAP_50",     "mAP@0.5"),
    ("mAP_50_95",  "mAP@0.5:0.95"),
    ("mAP_small",  "mAP_small"),
    ("mAP_medium", "mAP_medium"),
    ("mAP_large",  "mAP_large"),
    ("AR_small",   "AR_small"),
]

# Pairwise comparisons. A valid p-value needs BOTH sides n>=2; single-seed controls
# (E25/E18/E17) fall to a point-estimate row ("n<2"). t-test is run on each gated metric.
PAIRS = [
    ("E16_mca_scale_assign_hi32_1280", "E01_baseline_1280",              "v8 headline: hero vs baseline"),
    ("E16_mca_scale_assign_hi32_1280", "E02_p2_1280",                    "v8: hero vs +P2 (MCA+SA step)"),
    ("E02_p2_1280",                    "E01_baseline_1280",              "v8: +P2 step vs baseline"),
    ("E24_yolo11m_p2_mca_scale_1280",  "E18_yolov11m_baseline_b1_1280",  "v11 GO: hero vs fair baseline E25"),
    ("E16_mca_scale_assign_hi32_1280", "E17_p2_mca_p5trans_scale_1280",  "neg. ablation: hero vs +P5T"),
]
GATED = [("mAP_50_95", "mAP@0.5:0.95"), ("mAP_small", "mAP_small")]


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def welch_t(xs, ys):
    """Welch's t-test (unequal variance). Returns (t, p_two_sided).

    Uses scipy.stats.ttest_ind(equal_var=False) when available (exact); otherwise
    a crude normal-CDF approximation adequate only for sanity-check reporting.
    """
    if len(xs) < 2 or len(ys) < 2:
        return float("nan"), float("nan")
    if _scipy_stats is not None:
        res = _scipy_stats.ttest_ind(xs, ys, equal_var=False)
        return float(res.statistic), float(res.pvalue)
    mx, sx = mean_std(xs)
    my, sy = mean_std(ys)
    nx, ny = len(xs), len(ys)
    se = math.sqrt(sx * sx / nx + sy * sy / ny)
    if se == 0:
        return float("inf"), 0.0
    t = (mx - my) / se
    df = min(nx, ny) - 1
    z = t * math.sqrt(max(1.0, df / max(1, df - 2)))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return t, p


def collect():
    """{stem: {metric_key: [val_per_seed, ...]}} from cocoapi bucket exports."""
    runs = ROOT / "runs"
    out = defaultdict(lambda: defaultdict(list))
    for stem, _disp, _bb in EXPS:
        candidates = [runs / f"{stem}_eval_test_buckets"]  # seed 42
        for d in runs.iterdir():
            if d.is_dir() and re.match(rf"^{re.escape(stem)}_s(\d+)_eval_test_buckets$", d.name):
                candidates.append(d)
        for d in candidates:
            mj = d / "metrics.json"
            if not mj.exists():
                continue
            try:
                row = json.loads(mj.read_text(encoding="utf-8"))
            except Exception:
                continue
            for key, _ in METRICS:
                if key in row and row[key] is not None:
                    out[stem][key].append(float(row[key]))
    return out


def main():
    out_dir = ROOT / "runs" / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect()
    if not data:
        print("No *_eval_test_buckets/metrics.json found. Run eval_size_buckets.py first.")
        return

    disp = {stem: d for stem, d, _ in EXPS}
    bb = {stem: b for stem, _, b in EXPS}

    # ── mean ± std table ──
    md = ["### Multi-seed test-split metrics (mean ± std) — COCO protocol, imgsz=1280", ""]
    md.append("| Experiment | Backbone | n | " + " | ".join(l for _, l in METRICS) + " |")
    md.append("|" + "---|" * (3 + len(METRICS)))

    csv = ["exp,backbone,n_seeds," + ",".join(f"{l}_mean,{l}_std" for _, l in METRICS)]
    summary = {}
    for stem, _d, _b in EXPS:
        per_metric = data.get(stem, {})
        n = max((len(v) for v in per_metric.values()), default=0)
        cells, csv_cells = [], []
        for key, _ in METRICS:
            vals = per_metric.get(key, [])
            m, s = mean_std(vals)
            if not vals:
                cells.append("—")
            elif len(vals) < 2:
                cells.append(f"{m:.4f}")
            else:
                cells.append(f"{m:.4f} ± {s:.4f}")
            csv_cells.append(f"{m:.4f},{s:.4f}" if vals else ",")
        md.append(f"| {disp[stem]} | {bb[stem]} | {n} | " + " | ".join(cells) + " |")
        csv.append(f"{stem},{bb[stem]},{n}," + ",".join(csv_cells))
        summary[stem] = per_metric

    (out_dir / "seed_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "seed_stats.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved -> {out_dir / 'seed_stats.md'}")
    print(f"Saved -> {out_dir / 'seed_stats.csv'}")

    # ── pairwise Welch t-tests on the two gated metrics ──
    backend = "scipy.stats.ttest_ind(equal_var=False)" if _scipy_stats else "normal-CDF approx (install scipy for exact p)"
    sig_md = [f"### Pairwise Welch's t-test — engine: {backend}", ""]
    sig_md.append("| Comparison | metric | mean_A | mean_B | Δ (pp) | t | p | sig? |")
    sig_md.append("|---|---|---|---|---|---|---|---|")
    for a, b, note in PAIRS:
        for key, klabel in GATED:
            xs = summary.get(a, {}).get(key, [])
            ys = summary.get(b, {}).get(key, [])
            mx, _ = mean_std(xs) if xs else (float("nan"), 0)
            my, _ = mean_std(ys) if ys else (float("nan"), 0)
            if len(xs) < 2 or len(ys) < 2:
                d = f"{(mx - my) * 100:+.2f}" if xs and ys else "—"
                sig_md.append(f"| {note} | {klabel} | "
                              f"{mx:.4f} (n={len(xs)}) | {my:.4f} (n={len(ys)}) | {d} | — | n<2 | — |")
                continue
            t, p = welch_t(xs, ys)
            sig = "**yes**" if p < 0.05 else "no"
            sig_md.append(f"| {note} | {klabel} | {mx:.4f} | {my:.4f} | "
                          f"{(mx - my) * 100:+.2f} | {t:.3f} | {p:.4f} | {sig} |")
    sig_md.append("")
    sig_md.append("> Δ in percentage points. A valid p needs both sides n≥2; single-seed "
                  "controls (E25/E18/E17) show a point-estimate Δ only. Note per the lit "
                  "review that multi-seed t-tests are an above-bar bonus for this venue tier, "
                  "not a hard gate; report them where both arms have ≥2 seeds.")
    (out_dir / "seed_significance.md").write_text("\n".join(sig_md) + "\n", encoding="utf-8")
    print("\n".join(sig_md))
    print(f"\nSaved -> {out_dir / 'seed_significance.md'}")


if __name__ == "__main__":
    main()
