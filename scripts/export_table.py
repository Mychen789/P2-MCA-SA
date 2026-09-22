"""Emit two markdown + csv comparison tables: one for val split (training
best-epoch), one for test split (independent eval via val.py --split test).

Directory naming convention used to split:
  runs/EXX_<name>/                       training run        -> val table
  runs/EXX_<name>_eval_test/            standalone test eval -> test table
  runs/EXX_<name>_eval_test_softnms/    test + soft-nms      -> test table (variant)
  runs/EXX_<name>_eval_test_tta/        test + tta           -> test table (variant)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nmv.utils.metrics_io import read_best

_EVAL_SUFFIX = re.compile(r"_eval_(test|val)(_\w+)?$")


def imgsz_of(run_dir):
    """Authoritative imgsz: read from args.yaml of the corresponding training run; falls back to

    Audit H4: the old script did not parse the resolution and would mix 960 and 1280 evaluations
    """
    name = run_dir.name
    train_name = _EVAL_SUFFIX.sub("", name)         # strip _eval_test[...] / _eval_val[...]
    args = run_dir.parent / train_name / "args.yaml"
    if args.exists():
        m = re.search(r"^imgsz:\s*(\d+)", args.read_text(encoding="utf-8", errors="ignore"), re.M)
        if m:
            return int(m.group(1))
    return 1280 if "_1280" in name else 960          # name-based fallback


def collect_rows(dirs):
    rows = []
    for d in dirs:
        m = read_best(d / "results.csv")
        if m is None:
            continue
        rows.append((d.name, imgsz_of(d), m))
    # sort by (resolution, name) so that equal resolutions group together
    return sorted(rows, key=lambda r: (r[1], r[0]))


def emit_table(rows, out_md, out_csv, title, has_epoch=True):
    if not rows:
        print(f"[skip] no rows for {title}")
        return
    sample = rows[0][2]
    map_key = next((k for k in sample if "mAP50" in k and "95" not in k), None)
    map95_key = next((k for k in sample if "mAP50-95" in k or "mAP50_95" in k), None)
    p_key = next((k for k in sample if k.startswith("metrics/precision")), None)
    r_key = next((k for k in sample if k.startswith("metrics/recall")), None)

    # imgsz column: force every row to state its train/eval resolution, so cross-resolution
    hdr = "| # | Experiment | imgsz | " + ("epoch | " if has_epoch else "") + "P | R | mAP@0.5 | mAP@0.5:0.95 |"
    sep = "|---|------------|-------|" + ("-------|" if has_epoch else "") + "---|---|---------|--------------|"
    csv_hdr = "idx,name,imgsz," + ("epoch," if has_epoch else "") + "P,R,mAP50,mAP50_95"
    md = [f"### {title}", "",
          "> WARNING: rows with different imgsz **must not be compared**; their train/eval resolutions differ. For the paper, use only a single-imgsz sub-table.",
          "", hdr, sep]
    csv = [csv_hdr]
    for i, (name, imgsz, m) in enumerate(rows):
        P = m.get(p_key, 0) if p_key else 0
        R = m.get(r_key, 0) if r_key else 0
        m50 = m.get(map_key, 0) if map_key else 0
        m95 = m.get(map95_key, 0) if map95_key else 0
        if has_epoch:
            e = m.get("epoch", "?")
            md.append(f"| {i} | {name} | {imgsz} | {e} | {P:.4f} | {R:.4f} | {m50:.4f} | {m95:.4f} |")
            csv.append(f"{i},{name},{imgsz},{e},{P:.4f},{R:.4f},{m50:.4f},{m95:.4f}")
        else:
            md.append(f"| {i} | {name} | {imgsz} | {P:.4f} | {R:.4f} | {m50:.4f} | {m95:.4f} |")
            csv.append(f"{i},{name},{imgsz},{P:.4f},{R:.4f},{m50:.4f},{m95:.4f}")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    out_csv.write_text("\n".join(csv) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved -> {out_md}\nSaved -> {out_csv}\n")


def main():
    runs = ROOT / "runs"
    out_dir = runs / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dirs = sorted([d for d in runs.glob("E*") if d.is_dir()])
    val_dirs = [d for d in all_dirs if "_eval_test" not in d.name and not d.name.startswith("E07_sahi")]
    test_dirs = [d for d in all_dirs if "_eval_test" in d.name]

    def emit_grouped(dirs, stem, title, has_epoch):
        rows = collect_rows(dirs)
        # merged table (carries the imgsz column; cross-resolution rows are marked and must not be mixed)
        emit_table(rows, out_dir / f"{stem}.md", out_dir / f"{stem}.csv", title, has_epoch=has_epoch)
        # clean sub-tables split by resolution (use a single-imgsz sub-table for the paper)
        for sz in sorted({r[1] for r in rows}):
            sub = [r for r in rows if r[1] == sz]
            emit_table(sub, out_dir / f"{stem}_{sz}.md", out_dir / f"{stem}_{sz}.csv",
                       f"{title} — imgsz={sz}", has_epoch=has_epoch)

    emit_grouped(
        val_dirs, "comparison_val",
        "Val split — training best-epoch metrics (model selection set)",
        has_epoch=True,
    )
    emit_grouped(
        test_dirs, "comparison_test",
        "Test split — independent evaluation (reportable in paper)",
        has_epoch=False,
    )


if __name__ == "__main__":
    main()
