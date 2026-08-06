from pathlib import Path


def is_completed(run_dir):
    """Check if training actually completed (all epochs or early stopping).

    Priority order:
      1. last.pt has epoch=-1 → ultralytics strip_optimizer ran = training
         completed (most authoritative; this is exactly what ultralytics writes
         when its own _do_train loop hits all epochs OR the EarlyStopping
         stopper triggers).
      2. Fallback: compare last_epoch / best_epoch in results.csv against
         configured epochs / patience from args.yaml. Note: read_best uses
         mAP50, while ultralytics' EarlyStopping uses a fitness combination,
         so this fallback can disagree with the strip marker — hence why
         (1) is checked first.
    """
    d = Path(run_dir)
    if not (d / "weights" / "best.pt").exists() or not (d / "results.csv").exists():
        return False

    # (1) Authoritative ultralytics completion marker
    last_pt = d / "weights" / "last.pt"
    if last_pt.exists():
        try:
            import torch
            ckpt = torch.load(last_pt, weights_only=False, map_location="cpu")
            if isinstance(ckpt, dict) and ckpt.get("epoch") == -1:
                return True
        except Exception:
            pass

    # (2) Fallback: read configured epochs / patience from ultralytics args.yaml
    epochs, patience = 150, 30  # safe defaults
    args_yaml = d / "args.yaml"
    if args_yaml.exists():
        try:
            import yaml
            with open(args_yaml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            epochs = cfg.get("epochs", epochs)
            patience = cfg.get("patience", patience)
        except Exception:
            pass

    # Read last epoch and best epoch from results.csv
    best = read_best(d / "results.csv")
    if best is None:
        return False

    with open(d / "results.csv", "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        return False

    header = [h.strip() for h in lines[0].split(",")]
    last_cells = [c.strip() for c in lines[-1].split(",")]
    try:
        epoch_idx = header.index("epoch")
        last_epoch = int(float(last_cells[epoch_idx]))
    except (ValueError, IndexError):
        return True  # can't parse — assume completed for safety

    best_epoch = int(best.get("epoch", 0))

    # All epochs done OR early stopping triggered
    return last_epoch >= epochs - 1 or (last_epoch - best_epoch >= patience)


def read_best(results_csv):
    """Return the row with max mAP@0.5 from a YOLOv8 results.csv."""
    p = Path(results_csv)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        return None
    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split(",")]
        if len(cells) != len(header):
            continue
        try:
            rows.append({k: float(v) if k != "epoch" else int(float(v)) for k, v in zip(header, cells)})
        except ValueError:
            continue
    if not rows:
        return None
    map_key = next((k for k in header if "mAP50" in k and "95" not in k), None)
    if map_key is None:
        return rows[-1]
    return max(rows, key=lambda r: r.get(map_key, 0.0))
