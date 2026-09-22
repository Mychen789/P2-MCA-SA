"""Sequence-id audit (corrected version, 2026-06-27).

Sequence ids are parsed from the VisDrone filename `visdrone_<orig_split>_<seq>_<frame>_d_<id>.jpg`.
**The correction**: the sequence key is namespaced with orig_split (`train_<seq>` / `val_<seq>`), so that
sequence 0000072 of VisDrone-train and sequence 0000072 of VisDrone-val - entirely different videos - are
not collapsed onto one key. The old regex dropped the prefix into a non-capturing group, which both invented
train<->val/test leakage that did not exist and completely missed the real val<->test leakage.

This version reports all three pairwise sequence intersections (train<->val / train<->test / **val<->test**), showing:
  - whether train really is sequence-disjoint from val/test (zero-shot-vs-train)
  - whether val and test share sequences (the model-selection leakage channel)

Outputs:
  runs/_summary/sequence_audit.md
  runs/_summary/sequences_unseen.txt        (test/val images whose sequence is unseen in train)
"""
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(os.environ.get("NMV_DATA_ROOT", ROOT / "datasets" / "nmv_visdrone_3cls"))
OUT = ROOT / "runs" / "_summary"
OUT.mkdir(parents=True, exist_ok=True)

# filename: visdrone_<orig_split>_<seq>_<frame>_d_<id>.jpg
# group 1 = orig_split (train|val), group 2 = seq, group 3 = frame
SEQ_PAT = re.compile(
    r"visdrone_(train|val)_([0-9]+)_([0-9]+)_d_[0-9]+", re.IGNORECASE
)


def parse(filename):
    """Return (namespaced_seq, frame) or (None, None). namespaced_seq looks like 'train_0000072'."""
    m = SEQ_PAT.match(Path(filename).stem)
    if not m:
        return None, None
    return f"{m.group(1).lower()}_{m.group(2)}", int(m.group(3))


def seq_of(filename):
    return parse(filename)[0]


def load_cache_seqs(cache_path):
    c = np.load(str(cache_path), allow_pickle=True).item()
    out = defaultdict(int)
    files = []
    for lbl in c.get("labels", []):
        im_file = lbl.get("im_file", "")
        s = seq_of(im_file)
        if s:
            out[s] += 1
            files.append(Path(im_file).name)
    return dict(out), files


def scan_dir_seqs(d):
    out = defaultdict(int)
    files = []
    if not d.exists():
        return {}, []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        s = seq_of(p.name)
        if s:
            out[s] += 1
            files.append(p.name)
    return dict(out), files


def frames_in_shared(dir_seqs, dir_files, shared):
    """Number of images in a split whose sequence falls inside the shared-sequence set."""
    return sum(1 for f in dir_files if seq_of(f) in shared)


def main():
    train_cache = DATA_ROOT / "labels" / "train.cache"
    val_dir = DATA_ROOT / "images" / "val"
    test_dir = DATA_ROOT / "images" / "test"

    train_seqs, train_files = load_cache_seqs(train_cache)
    val_seqs, val_files = scan_dir_seqs(val_dir)
    test_seqs, test_files = scan_dir_seqs(test_dir)

    T, V, S = set(train_seqs), set(val_seqs), set(test_seqs)

    print(f"train.cache: {len(train_files)} imgs / {len(T)} seq")
    print(f"val/       : {len(val_files)} imgs / {len(V)} seq")
    print(f"test/      : {len(test_files)} imgs / {len(S)} seq")

    # the three pairwise sequence intersections
    tv = T & V
    ts = T & S
    vs = V & S
    print(f"\nSequence intersections:")
    print(f"  train & val  = {len(tv)} sequence(s)")
    print(f"  train & test = {len(ts)} sequence(s)")
    print(f"  val   & test = {len(vs)} sequence(s)  (model-selection leakage channel)")

    # frames covered by the shared sequences
    val_in_vs = frames_in_shared(val_seqs, val_files, vs)
    test_in_vs = frames_in_shared(test_seqs, test_files, vs)
    test_in_ts = frames_in_shared(test_seqs, test_files, ts)

    # check the source prefixes (train should be all 'train_*', val/test all 'val_*')
    src = lambda seqs: sorted({s.split("_")[0] for s in seqs})
    print(f"\nSource prefixes: train={src(T)}  val={src(V)}  test={src(S)}")

    md = [
        "# Sequence-id audit (corrected version, 2026-06-27)",
        "",
        "> The old regex discarded the orig_split prefix, so identically numbered sequences of VisDrone-train and VisDrone-val were merged,",
        "> which fabricated train<->val/test leakage and hid the real val<->test leakage. This version counts by the `orig_split_seq` namespace.",
        "",
        "## Size",
        "",
        "| split | images | sequences | sequence source prefix |",
        "|---|---:|---:|---|",
        f"| train | {len(train_files)} | {len(T)} | {','.join(src(T))} |",
        f"| val | {len(val_files)} | {len(V)} | {','.join(src(V))} |",
        f"| test | {len(test_files)} | {len(S)} | {','.join(src(S))} |",
        "",
        "## Sequence intersections (real leakage check)",
        "",
        "| pair | shared sequences | frames affected | conclusion |",
        "|---|---:|---:|---|",
        f"| train ∩ test | **{len(ts)}** | {test_in_ts}/{len(test_files)} | "
        + ("**train and test are sequence-disjoint; zero-shot-vs-train holds**" if len(ts) == 0 else "WARNING: training leakage") + " |",
        f"| train ∩ val | {len(tv)} | — | "
        + ("none" if len(tv) == 0 else "WARNING") + " |",
        f"| val ∩ test | **{len(vs)}** | val {val_in_vs}/{len(val_files)}, test {test_in_vs}/{len(test_files)} | "
        + ("none" if len(vs) == 0 else "WARNING: **model-selection leakage: best.pt is chosen by early stopping on val, and val shares sequences with test, so the headline test numbers carry selection bias and this must be disclosed**") + " |",
        "",
        "## Conclusion",
        "",
        f"- **train is sequence-disjoint from test** ({len(ts)}), so the paper's claim that test is sequence-disjoint from train / zero-shot-vs-train holds"
        + (", and more strongly than the old wording (the old buggy figure implied about 5% shared; it is in fact 0%)." if len(ts) == 0 else "."),
        f"- **val and test share {len(vs)}/{len(S)} sequences**, and {test_in_vs}/{len(test_files)} test frames fall inside those shared sequences."
        + " Because best.pt is chosen by early stopping on val, this is a val->test model-selection leak and **must be disclosed in the paper or removed by re-splitting**.",
        "- The 46/49 unseen sequences and 94.4%/95.6% shares in the old `sequence_audit.md` were artefacts of the regex bug and **should be withdrawn**;",
        "  the correct share of test sequences unseen in train is 100% (no test sequence appears in train).",
    ]
    (OUT / "sequence_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # test/val images whose sequence is unseen in train (after the fix, this should be all of them)
    unseen = [f"test/{f}" for f in test_files if seq_of(f) not in T]
    unseen += [f"val/{f}" for f in val_files if seq_of(f) not in T]
    (OUT / "sequences_unseen.txt").write_text("\n".join(unseen) + "\n", encoding="utf-8")

    print(f"\n[OK] -> {OUT/'sequence_audit.md'}")
    print(f"[OK] -> {OUT/'sequences_unseen.txt'} ({len(unseen)} image(s) unseen relative to train)")


if __name__ == "__main__":
    main()
