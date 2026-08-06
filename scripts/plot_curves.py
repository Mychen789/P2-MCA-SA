"""Overlay mAP@0.5 and box loss curves across all completed experiments."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nmv.utils.viz import plot_curves


def main():
    runs = ROOT / "runs"
    out_dir = runs / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    exps = sorted([d for d in runs.glob("E*") if d.is_dir() and (d / "results.csv").exists()])
    pairs = [(d.name, str(d)) for d in exps]
    if not pairs:
        print("No runs with results.csv found.")
        return

    plot_curves(
        pairs,
        metric_keys=["metrics/mAP50(B)", "metrics/mAP50"],
        out_path=str(out_dir / "curves_map50.png"),
        title="mAP@0.5 over epochs (val)",
        ylabel="mAP@0.5",
    )
    plot_curves(
        pairs,
        metric_keys=["train/box_loss"],
        out_path=str(out_dir / "curves_box_loss.png"),
        title="Box regression loss over epochs (train)",
        ylabel="box_loss",
    )
    print(f"Saved -> {out_dir / 'curves_map50.png'}")
    print(f"Saved -> {out_dir / 'curves_box_loss.png'}")


if __name__ == "__main__":
    main()
