"""确定性、序列无交集的数据划分工具（2026-06-27）。

解决审计 H3（划分无脚本/无 seed/无方法记录）与 H1（val↔test 泄漏）。
按 VisDrone 文件名 `visdrone_<orig_split>_<seq>_<frame>_d_<id>.jpg` 的
**带前缀命名空间序列号** `<orig_split>_<seq>` 分组，保证同一视频序列的所有帧
整体落入同一输出 split —— 任何两个输出 split 之间序列交集为 0。

分配是**完全确定性**的：序列按 (帧数降序, 键名升序) 排序后，逐个贪心放入
"当前最欠额"的 split（按目标帧数比例）。无随机数，结果可复现、可入版本控制。

子命令
------
make    从若干图像目录（pool）生成序列无交集的 N 路划分清单（stems）。
        典型用法（Track C：把现 val+test 重分为干净 val'/test'）:
          python scripts/make_split.py make \
            --pool D:/nmv_visdrone_3cls/images/val D:/nmv_visdrone_3cls/images/test \
            --names val test --ratios 0.5 0.5 \
            --out D:/nmv_visdrone_3cls/splits_clean

verify  检查若干已存在的 split 目录之间的序列交集（复用审计逻辑，做 CI 守门）。
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
    return f"{m.group(1).lower()}_{m.group(2)}"   # namespaced 序列键


def collect(dirs):
    """返回 {seq_key: [stems...]}（按 stem 升序），与未能解析的 stem 列表。"""
    groups = defaultdict(list)
    unparsed = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            print(f"[warn] 目录不存在: {d}", file=sys.stderr)
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
    """确定性贪心：序列(帧数降序,键升序) → 放入当前最欠额的 split。返回 {name: [stems]}。"""
    total = sum(len(v) for v in groups.values())
    targets = {n: r / sum(ratios) * total for n, r in zip(names, ratios)}
    out = {n: [] for n in names}
    cur = {n: 0 for n in names}
    order = sorted(groups.keys(), key=lambda k: (-len(groups[k]), k))
    for k in order:
        # 选"已分配/目标"比例最低者；并列取 names 顺序
        pick = min(names, key=lambda n: (cur[n] / targets[n] if targets[n] else 1e9, names.index(n)))
        out[pick].extend(groups[k])
        cur[pick] += len(groups[k])
    return out


def seqset(stems):
    return {parse(s) for s in stems if parse(s)}


def cmd_make(args):
    groups, unparsed = collect(args.pool)
    if unparsed:
        print(f"[warn] {len(unparsed)} 个文件名无法解析序列（已忽略）", file=sys.stderr)
    assert len(args.names) == len(args.ratios), "names 与 ratios 数量须一致"
    parts = assign(groups, args.names, args.ratios)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in args.names:
        (out / f"{n}.txt").write_text("\n".join(sorted(parts[n])) + "\n", encoding="utf-8")

    print(f"总序列 {len(groups)} / 总图像 {sum(len(v) for v in groups.values())}")
    for n in args.names:
        print(f"  {n}: {len(parts[n])} 图 / {len(seqset(parts[n]))} 序列  → {out/(n+'.txt')}")
    # 守门：两两序列交集必须为 0
    bad = 0
    for a, b in combinations(args.names, 2):
        inter = seqset(parts[a]) & seqset(parts[b])
        if inter:
            bad += 1
            print(f"  [FAIL] {a} ∩ {b} = {len(inter)} 序列泄漏: {sorted(inter)[:5]}...")
    print("[OK] 所有 split 序列两两无交集 ✓" if bad == 0 else f"[FAIL] {bad} 对存在序列交集")
    return 0 if bad == 0 else 1


def cmd_verify(args):
    named = {}
    for d in args.dirs:
        name = Path(d).name
        g, _ = collect([d])
        named[name] = set(g.keys())
        nimg = sum(len(v) for v in g.values())
        print(f"{name}: {nimg} 图 / {len(g)} 序列  来源前缀={sorted({s.split('_')[0] for s in g})}")
    print("\n序列交集矩阵:")
    leak = 0
    for a, b in combinations(named, 2):
        inter = named[a] & named[b]
        flag = "" if not inter else "  ⚠️ 泄漏"
        if inter:
            leak += 1
        print(f"  {a} ∩ {b} = {len(inter)} 序列{flag}")
    print("\n[OK] 无跨 split 序列交集 ✓" if leak == 0 else f"\n[WARN] {leak} 对存在序列交集，须披露或重划分")
    return 0


def main():
    ap = argparse.ArgumentParser(description="确定性序列无交集划分/校验工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="生成序列无交集的 N 路划分清单")
    m.add_argument("--pool", nargs="+", required=True, help="图像目录（可多个）")
    m.add_argument("--names", nargs="+", required=True, help="输出 split 名，如 val test")
    m.add_argument("--ratios", nargs="+", type=float, required=True, help="帧数目标比例")
    m.add_argument("--out", required=True, help="输出目录（写 <name>.txt 清单）")
    m.set_defaults(func=cmd_make)

    v = sub.add_parser("verify", help="校验已存在 split 目录间的序列交集")
    v.add_argument("--dirs", nargs="+", required=True, help="若干 split 图像目录")
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
