"""论文优化版 (imgsz=1280) 的单入口编排器 —— 在 VS Code 跑这一个脚本即可。

用法 (PowerShell):
  conda activate yolov8
  cd P2-MCA-SA
  python scripts/sanity_check.py        # 先确认 patch/解析 OK
  python scripts/run_paper.py           # ★跑完整优先级队列 (可随时 Ctrl+C, 再跑自动续训)

按优先级分阶段执行 (越靠前越关键, --until 可控制跑到哪个阶段就停):
  headline    : E08(P2+MCA) 与 E01(baseline) 各 1 个 seed=42 重训 → test 评估 → SAHI/分桶
  ablation    : 追加 E02(+P2) seed=42, 凑齐 baseline / +P2 / +P2+MCA 最小公平消融
  seeds       : E08 与 E01 各补 seed=1,7 (共 3 seed), 出 mean±std + 显著性
  components  : 同一 mca.yaml 上跑 MCA 组件消融 EMA-only / CoordAtt-only (NMV_MCA_MODE)

设计要点:
  - 所有 1280 新跑带 `_1280[_sN]` run 后缀, 不覆盖旧 960 结果 (留作分辨率消融)。
  - 训练交给 train.py (自带崩溃重试 + 断点续训)；本脚本对 eval/SAHI/分桶做幂等跳过。
  - 评估一律 imgsz=1280 (与训练分辨率一致, 否则整张表作废)。
  - SAHI 一律 conf=0.01, slice=640 (conf 太高会截断 PR 曲线、压低 mAP_small)。

可随时停: Ctrl+C 后重跑 `python scripts/run_paper.py` 会从断点继续 (train.py 的
early_stop_resume patch 保证)。每个 train run @1280/batch2 约 56h (smoke 实测 22.4min/epoch)。

其它常用参数:
  --until headline|ablation|seeds|components|all   跑到哪个阶段为止 (默认 all)
  --batch N         覆盖训练 batch (默认用 train.py 的 HP=2; 显存够可试 3)
  --smoke           只跑 2-epoch 烟雾测试 (验证当前 batch 在 1280 下不 OOM) 然后退出
  --dry-run         只打印队列与累计工时估算, 不真正执行
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN = str(ROOT / "scripts" / "train.py")
VAL = str(ROOT / "scripts" / "val.py")
SAHI = str(ROOT / "scripts" / "sahi_predict.py")
BUCKETS = str(ROOT / "scripts" / "eval_size_buckets.py")
RUNS = ROOT / "runs"

PHASE_RANK = {"headline": 1, "ablation": 2, "seeds": 3, "components": 4, "all": 99}

# train.py EXPERIMENTS 里的 idx → 基础 run 名 (apply_run_suffix 会在其后接 NMV_RUN_SUFFIX)
EXP_BASENAME = {1: "E01_baseline", 2: "E02_p2", 8: "E08_p2_mca"}


def banner(text, ch="#"):
    print("\n" + ch * 80 + f"\n  {text}\n" + ch * 80, flush=True)


def run_name(idx, suffix):
    return EXP_BASENAME[idx] + suffix


def _run(cmd, extra_env=None, label=""):
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    t0 = datetime.now()
    print(f"  $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    dt = datetime.now() - t0
    print(f"  -> rc={rc}  elapsed {dt}  [{label}]", flush=True)
    return rc


def do_train(idx, suffix, seed=42, mca_mode=None, batch=None, dry=False):
    """重训一个实验。train.py 自带 is_completed 跳过 + 崩溃重试 + resume。"""
    name = run_name(idx, suffix)
    extra = {"NMV_RUN_SUFFIX": suffix}
    if seed != 42:
        extra["NMV_SEED"] = seed
    if mca_mode:
        extra["NMV_MCA_MODE"] = mca_mode
    if batch:
        extra["NMV_BATCH"] = batch
    banner(f"TRAIN  {name}   (idx={idx} seed={seed}"
           + (f" mca_mode={mca_mode}" if mca_mode else "") + ")")
    if dry:
        print(f"  [dry-run] would train {name}")
        return name
    _run([PY, TRAIN, "--only", str(idx)], extra_env=extra, label=f"train {name}")
    return name


def do_eval(name, dry=False):
    """test 集评估 @imgsz=1280。幂等: 已有 results.csv 则跳过。"""
    out = RUNS / f"{name}_eval_test"
    if (out / "results.csv").exists():
        print(f"  [skip eval] {out.name} 已存在")
        return
    if dry:
        print(f"  [dry-run] would eval {name} @1280")
        return
    _run([PY, VAL, "--run", name, "--split", "test", "--imgsz", "1280", "--batch", "2"],
         label=f"eval {name}")


def do_sahi(name, dry=False):
    """SAHI 切片推理 (slice=640, conf=0.01)。幂等: 已有 metrics.json 则跳过。"""
    out = RUNS / f"E07_sahi_{name}_test"
    if (out / "metrics.json").exists():
        print(f"  [skip SAHI] {out.name} 已存在")
        return
    if dry:
        print(f"  [dry-run] would SAHI {name}")
        return
    _run([PY, SAHI, "--run", name, "--split", "test",
          "--slice", "640", "--overlap", "0.2", "--conf", "0.01"], label=f"sahi {name}")


def do_buckets(name, dry=False):
    """COCO 分桶 mAP_small/medium/large (非切片, conf=0.001)。幂等跳过。"""
    out = RUNS / f"{name}_eval_test_buckets"
    if (out / "metrics.json").exists():
        print(f"  [skip buckets] {out.name} 已存在")
        return
    if dry:
        print(f"  [dry-run] would size-bucket eval {name}")
        return
    _run([PY, BUCKETS, "--run", name, "--split", "test", "--imgsz", "1280"],
         label=f"buckets {name}")


def smoke_test(batch, dry=False):
    """2-epoch 烟雾测试: 在 1280 下用给定 batch 跑 E08(最重的 +P2+MCA), 验证不 OOM。"""
    banner(f"SMOKE TEST  E08 @1280 batch={batch or '默认(2)'}  2 epochs", "!")
    if dry:
        print("  [dry-run] would run 2-epoch smoke test"); return
    extra = {"NMV_RUN_SUFFIX": "_1280_smoke", "NMV_EPOCHS": "2"}
    if batch:
        extra["NMV_BATCH"] = batch
    rc = _run([PY, TRAIN, "--only", "8", "--force"], extra_env=extra, label="smoke")
    if rc == 0:
        print("\n[OK] 烟雾测试通过 —— 该 batch 在 1280 下不 OOM, 可以开始正式队列。")
        print("     正式跑请去掉 --smoke: python scripts/run_paper.py")
    else:
        print("\n[FAIL] 烟雾测试失败 (大概率 OOM)。把 batch 调小 (--batch 2) 再试。")
    return rc


def main():
    p = argparse.ArgumentParser(description="论文优化版 (imgsz=1280) 单入口编排器")
    p.add_argument("--until", choices=list(PHASE_RANK), default="all",
                   help="跑到哪个阶段为止 (默认 all)")
    p.add_argument("--batch", type=int, default=None,
                   help="覆盖训练 batch (默认用 train.py HP=2)")
    p.add_argument("--smoke", action="store_true", help="只跑 2-epoch 烟雾测试后退出")
    p.add_argument("--dry-run", action="store_true", help="只打印队列, 不执行")
    args = p.parse_args()

    batch = args.batch
    dry = args.dry_run

    if args.smoke:
        smoke_test(batch, dry=dry)
        return

    rank = PHASE_RANK[args.until]
    t_global = datetime.now()
    banner(f"PAPER QUEUE @imgsz=1280  until={args.until}  batch={batch or 'HP默认(2)'}"
           + ("  [DRY-RUN]" if dry else ""))
    print("  每 train run @1280/batch2 约 28h；eval ~10min；SAHI ~30min。")
    print("  可随时 Ctrl+C, 再跑本脚本自动续训。\n")
    print("  阶段累计工时参考: headline≈2.4天 | +ablation≈3.6天 | +3seed≈7天 | +components≈10天")

    # ---------------- Phase 1: headline (E08 + E01, seed 42) ----------------
    if rank >= PHASE_RANK["headline"]:
        banner("PHASE 1 / headline  (E08 P2+MCA  vs  E01 baseline, seed=42)", "=")
        e08 = do_train(8, "_1280", seed=42, batch=batch, dry=dry)
        e01 = do_train(1, "_1280", seed=42, batch=batch, dry=dry)
        for n in (e08, e01):
            do_eval(n, dry=dry); do_buckets(n, dry=dry); do_sahi(n, dry=dry)
        print("\n[GO 点 1] headline 完成: 1280 下 baseline vs P2+MCA 的标准 mAP / "
              "mAP_small(SAHI+分桶) / 每类 AP 已就绪 —— 单 seed 可投 Table 1。")

    # ---------------- Phase 2: ablation middle row (E02 +P2) ----------------
    if rank >= PHASE_RANK["ablation"]:
        banner("PHASE 2 / ablation  (E02 +P2, seed=42 —— 拆分 P2头 与 MCA 各自贡献)", "=")
        e02 = do_train(2, "_1280", seed=42, batch=batch, dry=dry)
        do_eval(e02, dry=dry); do_buckets(e02, dry=dry); do_sahi(e02, dry=dry)
        print("\n[GO 点 2] 最小公平消融完成: baseline / +P2 / +P2+MCA 三行同配方。")

    # ---------------- Phase 3: multi-seed (E08, E01 × seed 1,7) -------------
    if rank >= PHASE_RANK["seeds"]:
        banner("PHASE 3 / seeds  (E08 与 E01 各补 seed=1,7, 共 3 seed)", "=")
        for seed in (1, 7):
            for idx in (8, 1):
                n = do_train(idx, f"_1280_s{seed}", seed=seed, batch=batch, dry=dry)
                do_eval(n, dry=dry)
        print("\n[GO 点 3 / 推荐停点] 3 seed 齐. 跑 aggregate_seeds.py 出 mean±std + 显著性。")

    # ---------------- Phase 4: MCA component ablation -----------------------
    if rank >= PHASE_RANK["components"]:
        banner("PHASE 4 / components  (同 mca.yaml: EMA-only / CoordAtt-only)", "=")
        n_ema = do_train(8, "_1280_emaonly", seed=42, mca_mode="ema", batch=batch, dry=dry)
        do_eval(n_ema, dry=dry); do_buckets(n_ema, dry=dry)
        n_ca = do_train(8, "_1280_caonly", seed=42, mca_mode="ca", batch=batch, dry=dry)
        do_eval(n_ca, dry=dry); do_buckets(n_ca, dry=dry)
        print("\n[完成] 组件消融: EMA-only / CoordAtt-only / MCA-full(=E08_p2_mca_1280) 三者对照。")

    banner(f"DONE  until={args.until}  total elapsed {datetime.now() - t_global}")
    print("  生成论文素材 (训练/评估都跑完后):")
    print("    python scripts/export_table.py        # 主对比表 (会读 *_1280*_eval_test)")
    print("    python scripts/aggregate_seeds.py     # 3 seed mean±std + 显著性")
    print("    python scripts/per_class_ap.py        # 每类 AP")
    print("    python scripts/plot_curves.py / plot_pareto.py / plot_detections.py")
    print(f"\n  输出根目录: {RUNS}")


if __name__ == "__main__":
    main()
