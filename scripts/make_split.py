"""Deterministic, sequence-disjoint split tool (2026-06-27).

Addresses audit findings H3 (the split had no script, no seed and no recorded method) and H1 (val<->test leakage).
Images are grouped by the **namespaced sequence id** `<orig_split>_<seq>` parsed out of the VisDrone
filename `visdrone_<orig_split>_<seq>_<frame>_d_<id>.jpg`, so that every frame of one video sequence
lands in the same output split - the sequence intersection between any two output splits is 0.

The assignment is **fully deterministic**: sequences are sorted by (frame count descending, key ascending)
and greedily placed into whichever split is furthest below its target share of frames. No random numbers,

Subcommands
------
make    build a sequence-disjoint N-way split manifest (stems) from one or more image directories (pool).
        Typical use (Track C: re-partition the current val+test into clean val'/test'):
          python scripts/make_split.py make \
            --pool D:/nmv_visdrone_3cls/images/val D:/nmv_visdrone_3cls/images/test \
            --names val test --ratios 0.5 0.5 \
            --out D:/nmv_visdrone_3cls/splits_clean

verify  check the sequence intersection between existing split directories (reuses the audit logic, as a CI gate).
          python scripts/make_split.py verify \
            --dirs D:/nmv_visdrone_3cls/images/train \
                   D:/nmv_visdrone_3cls/images/val \
                   D:/nmv_visdrone_3cls/images/test
"""
import argparse
import re
import sys
from itertools import combinations
from pathlib import Path
from collections import defaultdict

SEQ_PAT = re.compile(r"visdrone_(train|val)_([0-9]+)_([0-9]+)_d_[0-9]+", re.IGNORECASE)
IMG_EXT = {".jpg", ".jpeg", ".png"}


def parse(stem):
    m = SEQ_PAT.match(stem)
    if not m:
        return None
    return f"{m.group(1).lower()}_{m.group(2)}"   # namespaced sequence key


def collect(dirs):
    """Return {seq_key: [stems...]} (stems ascending) and the list of stems that could not be parsed."""
    groups = defaultdict(list)
    unparsed = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            print(f"[warn] directory does not exist: {d}", file=sys.stderr)
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            k = parse(p.stem)
            (groups[k] if k else unparsed).append(p.stem if k else p.name)
    for k in groups:
        groups[k].sort()
    return dict(groups), unparsed


def assign(groups, names, ratios):
    """Deterministic greedy fill: sequences (frames desc, key asc) go to the split furthest below target. Returns {name: [stems]}."""
    total = sum(len(v) for v in groups.values())
    targets = {n: r / sum(ratios) * total for n, r in zip(names, ratios)}
    out = {n: [] for n in names}
    cur = {n: 0 for n in names}
    order = sorted(groups.keys(), key=lambda k: (-len(groups[k]), k))
    for k in order:
        # pick the lowest assigned/target ratio; ties broken by the order of names
        pick = min(names, key=lambda n: (cur[n] / targets[n] if targets[n] else 1e9, names.index(n)))
        out[pick].extend(groups[k])
        cur[pick] += len(groups[k])
    return out


def seqset(stems):
    return {parse(s) for s in stems if parse(s)}


def cmd_make(args):
    groups, unparsed = collect(args.pool)
    if unparsed:
        print(f"[warn] {len(unparsed)} filename(s) could not be parsed into a sequence (ignored)", file=sys.stderr)
    assert len(args.names) == len(args.ratios), "names and ratios must have the same length"
    parts = assign(groups, args.names, args.ratios)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in args.names:
        (out / f"{n}.txt").write_text("\n".join(sorted(parts[n])) + "\n", encoding="utf-8")

    print(f"sequences {len(groups)} / images {sum(len(v) for v in groups.values())}")
    for n in args.names:
        print(f"  {n}: {len(parts[n])} images / {len(seqset(parts[n]))} sequences  -> {out/(n+'.txt')}")
    # gate: every pairwise sequence intersection must be 0
    bad = 0
    for a, b in combinations(args.names, 2):
        inter = seqset(parts[a]) & seqset(parts[b])
        if inter:
            bad += 1
            print(f"  [FAIL] {a} & {b} = {len(inter)} leaked sequence(s): {sorted(inter)[:5]}...")
    print("[OK] all splits are pairwise sequence-disjoint" if bad == 0 else f"[FAIL] {bad} pair(s) share sequences")
    return 0 if bad == 0 else 1


def cmd_verify(args):
    named = {}
    for d in args.dirs:
        name = Path(d).name
        g, _ = collect([d])
        named[name] = set(g.keys())
        nimg = sum(len(v) for v in g.values())
        print(f"{name}: {nimg} images / {len(g)} sequences  source prefixes={sorted({s.split('_')[0] for s in g})}")
    print("\nSequence intersection matrix:")
    leak = 0
    for a, b in combinations(named, 2):
        inter = named[a] & named[b]
        flag = "" if not inter else "  WARNING: leakage"
        if inter:
            leak += 1
        print(f"  {a} & {b} = {len(inter)} sequence(s){flag}")
    print("\n[OK] no cross-split sequence intersection" if leak == 0 else f"\n[WARN] {leak} pair(s) share sequences; disclose or re-split")
    return 0


def main():
    ap = argparse.ArgumentParser(description="deterministic sequence-disjoint split / verification tool")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="build a sequence-disjoint N-way split manifest")
    m.add_argument("--pool", nargs="+", required=True, help="image directories (one or more)")
    m.add_argument("--names", nargs="+", required=True, help="output split names, e.g. val test")
    m.add_argument("--ratios", nargs="+", type=float, required=True, help="target share of frames")
    m.add_argument("--out", required=True, help="output directory (writes <name>.txt manifests)")
    m.set_defaults(func=cmd_make)

    v = sub.add_parser("verify", help="check the sequence intersection between existing split directories")
    v.add_argument("--dirs", nargs="+", required=True, help="split image directories")
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
