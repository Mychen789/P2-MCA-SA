"""序列号审计（修正版，2026-06-27）。

按 VisDrone 文件名 `visdrone_<orig_split>_<seq>_<frame>_d_<id>.jpg` 解析序列号。
**关键修正**：序列键带 orig_split 前缀命名空间（`train_<seq>` / `val_<seq>`），
避免 VisDrone-train 的 seq 0000072 与 VisDrone-val 的 seq 0000072（完全不同的视频）
被折叠成同一键。旧版正则把前缀丢进非捕获组，既凭空制造 train↔val/test 假泄漏、
又彻底漏检了真实的 val↔test 泄漏。

本版输出三对序列交集（train↔val / train↔test / **val↔test**），如实反映：
  - train 对 val/test 是否真的序列无交集（zero-shot-vs-train）
  - val 与 test 之间是否共享序列（模型选择泄漏渠道）

输出:
  runs/_summary/sequence_audit.md
  runs/_summary/sequences_unseen.txt        (相对 train 未见序列的 test/val 图像)
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

# 文件名: visdrone_<orig_split>_<seq>_<frame>_d_<id>.jpg
# 捕获组1 = orig_split (train|val), 组2 = seq, 组3 = frame
SEQ_PAT = re.compile(
    r"visdrone_(train|val)_([0-9]+)_([0-9]+)_d_[0-9]+", re.IGNORECASE
)


def parse(filename):
    """返回 (namespaced_seq, frame) 或 (None, None)。namespaced_seq = 'train_0000072'。"""
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
    """某 split 中落在 shared 序列集合内的图像数。"""
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

    # 三对序列交集
    tv = T & V
    ts = T & S
    vs = V & S
    print(f"\n序列交集:")
    print(f"  train ∩ val  = {len(tv)} 序列")
    print(f"  train ∩ test = {len(ts)} 序列")
    print(f"  val   ∩ test = {len(vs)} 序列  (模型选择泄漏渠道)")

    # 共享序列覆盖的帧数
    val_in_vs = frames_in_shared(val_seqs, val_files, vs)
    test_in_vs = frames_in_shared(test_seqs, test_files, vs)
    test_in_ts = frames_in_shared(test_seqs, test_files, ts)

    # 前缀来源核验（应当 train 全部来自 'train_*'，val/test 全部来自 'val_*'）
    src = lambda seqs: sorted({s.split("_")[0] for s in seqs})
    print(f"\n前缀来源: train={src(T)}  val={src(V)}  test={src(S)}")

    md = [
        "# 序列号审计（修正版，2026-06-27）",
        "",
        "> 旧版正则丢弃 orig_split 前缀，导致 VisDrone-train 与 VisDrone-val 的同号序列被误并，",
        "> 既伪造 train↔val/test 泄漏、又漏检真实的 val↔test 泄漏。本版按 `orig_split_seq` 命名空间统计。",
        "",
        "## 规模",
        "",
        "| split | 图像 | 序列 | 序列来源前缀 |",
        "|---|---:|---:|---|",
        f"| train | {len(train_files)} | {len(T)} | {','.join(src(T))} |",
        f"| val | {len(val_files)} | {len(V)} | {','.join(src(V))} |",
        f"| test | {len(test_files)} | {len(S)} | {','.join(src(S))} |",
        "",
        "## 序列交集（真实泄漏检查）",
        "",
        "| 序列对 | 共享序列数 | 受影响帧 | 结论 |",
        "|---|---:|---:|---|",
        f"| train ∩ test | **{len(ts)}** | {test_in_ts}/{len(test_files)} | "
        + ("**train-vs-test 序列无交集，zero-shot-vs-train 成立**" if len(ts) == 0 else "⚠️ 训练泄漏") + " |",
        f"| train ∩ val | {len(tv)} | — | "
        + ("无" if len(tv) == 0 else "⚠️") + " |",
        f"| val ∩ test | **{len(vs)}** | val {val_in_vs}/{len(val_files)}, test {test_in_vs}/{len(test_files)} | "
        + ("无" if len(vs) == 0 else "⚠️ **模型选择泄漏：best.pt 经 val 早停选取，val 与 test 同源 → headline test 含选择偏置，须披露**") + " |",
        "",
        "## 结论",
        "",
        f"- **train 对 test 序列无交集**（{len(ts)}），故论文「test 相对 train 为序列无交集 / zero-shot-vs-train」成立"
        + ("，且比旧表述更强（旧 buggy 数字暗示约 5% 共享，实为 0%）。" if len(ts) == 0 else "。"),
        f"- **val 与 test 共享 {len(vs)}/{len(S)} 序列**，{test_in_vs}/{len(test_files)} 张 test 帧落在共享序列内。"
        + "由于 best.pt 经 val 早停选取，这构成 val→test 模型选择泄漏，**必须在论文中披露或通过重划分消除**。",
        "- 旧 `sequence_audit.md` 的 46/49 未见序列、94.4%/95.6% 占比为 regex bug 产物，**应撤销**；",
        "  正确的「test 相对 train 未见序列」占比为 100%（全部 test 序列均不在 train）。",
    ]
    (OUT / "sequence_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # test/val 中相对 train 未见序列的图像（修正后应为全部）
    unseen = [f"test/{f}" for f in test_files if seq_of(f) not in T]
    unseen += [f"val/{f}" for f in val_files if seq_of(f) not in T]
    (OUT / "sequences_unseen.txt").write_text("\n".join(unseen) + "\n", encoding="utf-8")

    print(f"\n[OK] -> {OUT/'sequence_audit.md'}")
    print(f"[OK] -> {OUT/'sequences_unseen.txt'} ({len(unseen)} 张相对 train 未见)")


if __name__ == "__main__":
    main()
