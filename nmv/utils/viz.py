from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_curves(run_dirs, metric_keys, out_path, title=None, ylabel=None):
    """Overlay one metric across multiple runs.

    run_dirs: list of (label, path-to-run-dir)
    metric_keys: ordered list of column names to try (first found wins, handles
                 column-name drift across ultralytics versions)
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    for label, d in run_dirs:
        csv = Path(d) / "results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        col = next((k for k in metric_keys if k in df.columns), None)
        if col is None:
            continue
        ax.plot(df["epoch"], df[col], label=label, linewidth=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel or (metric_keys[0] if metric_keys else "value"))
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
