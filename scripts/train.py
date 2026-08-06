"""Main training entry for the non-motor-vehicle small-target detection paper.

Each experiment is dispatched as an isolated subprocess so monkey-patches
(MPDIoU loss / Soft-NMS) cannot leak between experiments. The parent reads
results.csv after each child returns and prints a summary table at the end.

Experiments E01-E11 (unified condition: imgsz=960, epochs=150, patience=50, batch=4, seed=42).
- E01-E05: EMA-branch ablation chain (baseline → +P2 → +EMA → +MPDIoU → +GFPN)
- E06-E07: inference enhancements (Soft-NMS / TTA on E05 best.pt)
- E08-E11: ⭐ Two new innovations of this work (MCA + CAGFPN)
- SAHI inference handled separately by scripts/sahi_predict.py

Usage in VSCode terminal:
  cd P2-MCA-SA
  python scripts/train.py                 # ★裸跑 = 论文优化版 1280 主线 (PAPER_QUEUE):
                                          #   训练 baseline/+P2/+P2+MCA + 2 个补充 seed,
                                          #   每个练完自动 test 评估; 全部写入 _1280 新目录,
                                          #   不动旧 960 结果; 可随时 Ctrl+C, 再跑自动续训。
  python scripts/train.py --only 8        # (进阶) 只跑某个 EXPERIMENTS 实验 (旧 960 命名)
  python scripts/train.py --start 8       # (进阶) 从某 idx 开始跑旧 EXPERIMENTS 列表
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nmv  # triggers patch injection (env-var driven)
from nmv.utils.metrics_io import is_completed, read_best
from nmv.utils.data import ensure_data_yaml


# NMV_DATA 覆盖数据集 yaml（如 10 类锚点实验 configs/data/visdrone10.yaml），默认 3 类主数据集
DATA = ensure_data_yaml(Path(os.environ.get(
    "NMV_DATA", str(ROOT / "configs" / "data" / "nmv_visdrone_3cls.yaml"))))
MODELS = ROOT / "configs" / "models"
RUNS = ROOT / "runs"
WEIGHTS = ROOT / "weights"


EXPERIMENTS = [
    # ========= EMA 分支累积消融 E01-E05 (作为对比基线) =========
    dict(
        idx=1, name="E01_baseline",
        kind="train", cfg=None, weights="yolov8m.pt",
        env={},
        desc="YOLOv8m baseline (3 detection heads, CIoU, hard NMS)",
    ),
    dict(
        idx=2, name="E02_p2",
        kind="train", cfg=str(MODELS / "yolov8m-p2.yaml"), weights="yolov8m.pt",
        env={},
        desc="+P2 detection head (4 scales: P2/4, P3/8, P4/16, P5/32)",
    ),
    dict(
        idx=3, name="E03_p2_ema",
        kind="train", cfg=str(MODELS / "yolov8m-p2-ema.yaml"), weights="yolov8m.pt",
        env={},
        desc="+EMA attention (factor=4) on P2 features",
    ),
    dict(
        idx=4, name="E04_p2_ema_mpdiou",
        kind="train", cfg=str(MODELS / "yolov8m-p2-ema.yaml"), weights="yolov8m.pt",
        env={"NMV_IOU": "mpdiou"},
        desc="+MPDIoU regression loss (Ma, arXiv:2307.07662)",
    ),
    dict(
        idx=5, name="E05_p2_ema_mpdiou_gfpn",
        kind="train", cfg=str(MODELS / "yolov8m-p2-ema-gfpn.yaml"), weights="yolov8m.pt",
        env={"NMV_IOU": "mpdiou"},
        desc="EMA-branch full: P2 + EMA + MPDIoU + GFPN cross-scale skip",
    ),

    # ========= E05 推理增强 E06-E07 (val_only, 复用 E05 best.pt) =========
    dict(
        idx=6, name="E06_softnms",
        kind="val_only", source="E05_p2_ema_mpdiou_gfpn",
        env={"NMV_IOU": "mpdiou", "NMV_NMS": "soft"},
        desc="E05 best.pt + Soft-NMS at inference (no retrain)",
    ),
    dict(
        idx=7, name="E07_tta",
        kind="val_only", source="E05_p2_ema_mpdiou_gfpn",
        env={"NMV_IOU": "mpdiou"}, val_kwargs=dict(augment=True),
        desc="E05 best.pt + Test-Time Augmentation (multi-scale + flip)",
    ),
    # (Soft-NMS+TTA SAHI 在 scripts/sahi_predict.py 单独跑, 不进 EXPERIMENTS)

    # ========= 本工作 2 项新创新 E08-E11 (MCA + CAGFPN) =========
    dict(
        idx=8, name="E08_p2_mca",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca.yaml"), weights="yolov8m.pt",
        env={},
        desc="⭐ +MCA 多分支交叉注意力 (本工作: EMA × CoordAtt gated fusion)",
    ),
    dict(
        idx=9, name="E09_p2_mca_mpdiou",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca.yaml"), weights="yolov8m.pt",
        env={"NMV_IOU": "mpdiou"},
        desc="+MPDIoU (on MCA base)",
    ),
    dict(
        idx=10, name="E10_p2_mca_mpdiou_gfpn",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-gfpn.yaml"), weights="yolov8m.pt",
        env={"NMV_IOU": "mpdiou"},
        desc="+GFPN (MCA base, no context augmentation)",
    ),
    dict(
        idx=11, name="E11_full_mca_cagfpn",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-cagfpn.yaml"), weights="yolov8m.pt",
        env={"NMV_IOU": "mpdiou"},
        desc="⭐⭐ Full model: P2 + MCA + MPDIoU + CAGFPN (本工作 2 项新创新)",
    ),

    # ========= 外部对比基线 E12-E13 (不同尺寸 YOLOv8) =========
    # 用于论文 "size vs accuracy" Pareto 曲线，证明 E11 在效率/精度权衡上的位置
    # 注意: yolov8s.pt / yolov8l.pt 若 weights/ 下不存在，ultralytics 会自动联网下载
    # yolov8l 在 8GB GPU 上可能需要 NMV_BATCH=2 才能跑通 imgsz=960
    dict(
        idx=12, name="E12_yolov8s_baseline",
        kind="train", cfg=None, weights="yolov8s.pt",
        env={},
        desc="YOLOv8s baseline (smaller backbone for size-accuracy Pareto curve)",
    ),
    dict(
        idx=13, name="E13_yolov8l_baseline",
        kind="train", cfg=None, weights="yolov8l.pt",
        env={},
        desc="YOLOv8l baseline (larger backbone for size-accuracy Pareto curve)",
    ),

    # ========= MCA 隔离消融 E14 (mAP_large 修复实验) =========
    # 2026-05-19 诊断：E08/E11 的 mAP_large 暴跌源于 MCA 输出经 BU 路径下采样
    # 污染 P3/P4/P5 大目标特征。E14 仅改 yaml 第 20 行：BU 下采样从 pre-MCA P2
    # 取源 (idx=18) 而非 MCA 输出 (idx=-1=19)，把 MCA 局部化到 P2 检测头。
    # 目标：mAP_large 回到 ≥0.47 同时保留 mAP_small ≥0.215，总 mAP@0.5 ≥0.495。
    dict(
        idx=14, name="E14_full_mca_cagfpn_isolated",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-cagfpn-isolated.yaml"),
        weights="yolov8m.pt",
        env={"NMV_IOU": "mpdiou"},
        desc="⭐ MCA-isolated variant of E11: BU sources from pre-MCA P2 (recover mAP_large)",
    ),

    # ========= R2 防御反例实验 E15 (对称 MCA, 应表现更差) =========
    # 2026-05-19 设计：E14 基础上在 P5 头也加 MCA，验证审稿人 R2 攻击
    # "为什么不直接在 P5 也加一个 MCA" 的答案 —— 对称 MCA 因 MCA 设计本身
    # 面向密集小目标，应用于 P5 会过度抑制大目标边界回归所需的非中心激活。
    # 预期：AP_large 进一步退化、参数 +~1.5M、整体 mAP 不升反降，从而证明
    # E14 的非对称 MCA-L 设计是正确选择。
    dict(
        idx=15, name="E15_full_mca_cagfpn_p5mca",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-cagfpn-p5mca.yaml"),
        weights="yolov8m.pt",
        env={"NMV_IOU": "mpdiou"},
        desc="R2-defense: symmetric MCA on P5 (negative-result ablation for MCA-L)",
    ),

    # ========= 大目标修复 E16 (尺度感知标签分配) =========
    # 2026-05-22 诊断翻盘：mAP_large 暴跌的根因是 P2 头本身（E01 0.509→E02 0.303），
    # 大目标被分到无法回归它们的高分辨率 P2 anchor 上、并饿死 P5 头。E16 = E08(P2+MCA)
    # + size-range 标签分配（仅加上界：每个头只分配 max_side < hi_ratio*stride 的目标）。
    # 与 E08 干净对照，隔离"分配修复"对大目标的单项效果。成功判据：mAP_large 回 0.45+
    # 且 mAP_small 保住 ≥0.215。通过 NMV_SCALE_ASSIGN=1 启用 scale_aware_assign patch。
    dict(
        idx=16, name="E16_mca_scale_assign",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1"},
        desc="P1: P2+MCA + scale-aware label assignment (recover mAP_large)",
    ),

    # ========= E16 HI_RATIO=32 消融 (2026-05-28 加) =========
    # E16 (HI_RATIO=16) test bucket mAP_large=0.248，比 E08 (P2+MCA, large=0.301)
    # 反而退 5.3pp，证伪"上界标签分配能修大目标"。E16 hi32 放宽 P4 接受范围到
    # max_side < 32×16=512px，让中大目标也能在 P4 拿正样本，验证 HI_RATIO 是否
    # 关键。E16' (HI=32) 翻盘成功 (large 0.247→0.390, mAP@.5=0.520 全表第一)。
    dict(
        idx=17, name="E16_mca_scale_assign_hi32",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32"},
        desc="P1+: P2+MCA + scale-aware (HI_RATIO=32, 放宽 P4 至 ≤512px GT)",
    ),

    # ========= E17 P5 Transformer 全局分支 (2026-06-02 加, Neurocomputing 二区主英雄) =========
    # E16' (HI=32) 把 mAP_large 从 0.247 救到 0.390 但仍未追平 baseline 0.509。
    # 残余 -11.9pp 归因于 P5 特征质量本身：stride=32 CNN-only 在大目标上感受野
    # /全局上下文不足。E17 在 P5 (30×30 tokens @ imgsz=960) 上加 2 层 Transformer
    # encoder (dim=192, 4 heads), 与 CNN P5 门控残差融合 (proj_out 零初始化, 起步
    # 即恒等)。配 scale-aware (HI=32) 一起训, 目标 mAP@.5 > 0.520, mAP_large > 0.45,
    # 参数增量 < 0.5M, 8GB 不 OOM。论文主创新点①。
    dict(
        idx=18, name="E17_p2_mca_p5trans_scale",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-p5trans.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32"},
        desc="⭐ P2 + MCA + P5 Transformer + scale-aware (E17 二区主英雄)",
    ),

    # ========= E18 外部 SOTA 对比基线 YOLOv11m (2026-06-02 加) =========
    # 二区审稿人会问"为什么不跟近期 SOTA 比"。本 entry 用 ultralytics 自带 yolo11m.pt
    # (首次跑会自动从 Ultralytics 官网下载) 在同数据集 imgsz=960 batch=4 训练, 与 E01-E17
    # 完全对齐, 输出公平对比数字。预期 ~25h GPU。论文 Tab.6 SOTA 对比的核心数据来源。
    dict(
        idx=19, name="E18_yolov11m_baseline",
        kind="train", cfg=None, weights="yolo11m.pt",
        env={},
        desc="External SOTA baseline: YOLOv11m on NMV-SOD-3cls (paper Tab.6 SOTA cmp)",
    ),

    # ========= E17 ablation 套件 (2026-06-02 加, §5.10 ablation) =========
    # E17 主英雄 (dim=192, depth=2) 已是最小稳定配置. 论文 §5.10 ablation 需要扫:
    #   (a) dim ∈ {128, 192, 256} at depth=2 → E20 / E17 / E21
    #   (b) depth ∈ {1, 2, 4} at dim=192    → E22 / E17 / E23
    # 共 4 个新配置 + E17 共享中心. Gate ablation 通过 NMV_P5TRANS_MODE 环境变量 (full /
    # trans_only / cnn_only) 在 E17 上跑, 不需要新 yaml.
    dict(
        idx=20, name="E20_p5trans_d128_L2",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-p5trans-d128-L2.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32"},
        desc="ablation: P5 Transformer dim=128 depth=2 (smaller dim variant)",
    ),
    dict(
        idx=21, name="E21_p5trans_d256_L2",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-p5trans-d256-L2.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32"},
        desc="ablation: P5 Transformer dim=256 depth=2 (larger dim variant)",
    ),
    dict(
        idx=22, name="E22_p5trans_d192_L1",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-p5trans-d192-L1.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32"},
        desc="ablation: P5 Transformer dim=192 depth=1 (shallower variant)",
    ),
    dict(
        idx=23, name="E23_p5trans_d192_L4",
        kind="train", cfg=str(MODELS / "yolov8m-p2-mca-p5trans-d192-L4.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32"},
        desc="ablation: P5 Transformer dim=192 depth=4 (deeper variant)",
    ),

    # ========= E24 迁移验证 (2026-06-18 加): 把创新移植到 YOLOv11m 骨干 =========
    # E18 YOLOv11m 纯基线在整体 mAP + 小目标 mAP 上均反超 v8 系 hero (E16'), 且模型更小.
    # 根因: 这是 "旧骨干(v8m)+手工模块" vs "新骨干(v11m)" 的非同台对比. E24 = YOLOv11m +
    # P2 头 + MCA + 尺度感知分配(HI=32), 与 E18 在同一起跑线对比, 验证创新对最新骨干是否
    # 仍有正交增益. cfg 的 backbone 照抄官方 yolo11 → .load(yolo11m.pt) 加载 COCO 预训练 backbone.
    dict(
        idx=24, name="E24_yolo11m_p2_mca_scale",
        kind="train", cfg=str(MODELS / "yolo11m-p2-mca.yaml"), weights="yolo11m.pt",
        # NMV_BATCH=1: v11m(width1.0)+P2 高分辨率头 batch2@1280 forward 占满 7.18GB 后,
        #   标签分配(TaskAlignedAssigner; mosaic 多框 × P2 海量 anchor → align_metric 矩阵巨大)
        #   无 GPU 余量 → 频繁回退 CPU(GPU 空转 + 吃满系统 RAM)。8GB 卡只能 batch=1(峰值~4.3GB,
        #   TAL 全程 GPU)。代价: BN 单样本偏弱, 但 nbs=64 梯度累积保优化; 与 E18(batch2)对比时注明,
        #   想要 batch2 正式可比需 ≥16GB GPU。NMV_GPU_LIMIT_GB=7.5 保留(batch1 用不满, 无害)。
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32", "NMV_GPU_LIMIT_GB": "7.5", "NMV_BATCH": "1"},
        desc="⭐ 迁移验证: YOLOv11m + P2 + MCA + scale-aware(HI=32) vs E18 (8GB→batch1)",
    ),

    # ========= E25 公平 v11 基线 (2026-06-21 加): E18 的 batch=1 版本 =========
    # E24 因 8GB 只能 batch=1, 原 E18(idx19) 是 batch=2 → 跨 batch 对比不公平 (BN 单样本 +
    # 优化轨迹差异, 审稿必挑). E25 = YOLOv11m 纯基线 @batch=1, 与 E24 完全同条件
    # (imgsz=1280 / SGD / 150ep / 同增强), 仅消除 batch confound, 做成 apples-to-apples.
    # cfg=None + weights=yolo11m.pt → 加载完整 COCO 预训练 yolo11m 微调 (同原 E18, 只改 batch).
    # 不覆盖原 E18(batch2) 结果 (run 目录 _b1 后缀, 留作 batch 敏感性参考).
    # batch=1 等参数烤进 env → 用户裸跑 train.py 无需手设任何环境变量.
    dict(
        idx=25, name="E18_yolov11m_baseline_b1",
        kind="train", cfg=None, weights="yolo11m.pt",
        env={"NMV_BATCH": "1", "NMV_GPU_LIMIT_GB": "7.5"},
        desc="公平 v11 基线: YOLOv11m @batch=1 (与 E24 apples-to-apples; 仅消除 batch confound)",
    ),

    # ========= 关键补充消融 E26: P2 + SA（不含 MCA） =========
    # 用于隔离尺度感知分配本身的贡献，补齐 baseline / P2 / P2+SA / P2+MCA /
    # P2+MCA+SA 五行消融。HI_RATIO=32 与论文当前最终模型 E16' 保持一致。
    dict(
        idx=26, name="E26_p2_scale_assign_hi32",
        kind="train", cfg=str(MODELS / "yolov8m-p2.yaml"), weights="yolov8m.pt",
        env={"NMV_SCALE_ASSIGN": "1", "NMV_SCALE_HI_RATIO": "32"},
        desc="关键补充消融: P2 + scale-aware assignment (HI_RATIO=32, no MCA)",
    ),
]


HP = dict(
    epochs=150,
    imgsz=1280,   # 960→1280: 论文优化版。图像原生最高 1920×1080(中位 1360×765),960 在降采样;
                  #           1280 是小目标最大单项训练增益。旧 960 结果保留(带不同 run 后缀)做分辨率消融。
    batch=2,      # 4→2: imgsz 1280 激活随 (1280/960)^2=1.78x 增长,8GB 下 batch=4 必 OOM;batch=2 峰值~6.7GB 安全。
                  #      可用 NMV_BATCH=3 试(先 smoke test);BatchNorm 小 batch 噪声靠 mosaic/mixup+warmup 缓解,不用 SyncBN。
    patience=50,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.3,
    close_mosaic=15,
    workers=0,    # 2→0：Windows DataLoader 子进程 import torch 占 ~1GB 虚拟内存，长 run 累积地址空间碎片是 Phase A 中段 OOM 主因；retry 已验证 0 可跑通；cache='disk' 抵消 IO 损失
    cache='disk',
    amp=True,
    seed=42,
    device=0,
    verbose=True,
    plots=True,
)


MAX_ATTEMPTS = 3   # 允许的连续"零 epoch 进展"重试次数；用尽(真卡死)才 sys.exit(1)。只要有新 epoch 落盘即重置(见 dispatch_one 进度感知重试)


# 2026-05-28: 裸跑 train.py 时默认跳过的实验。各自的失效原因写在下面注释里。
# 这些实验仍然保留在 EXPERIMENTS 表内供历史追溯。需要单跑某个被默认跳过的实验，
# 用 `--only <idx>`（--only 会绕过 DEFAULT_SKIP，按用户意图强制执行）。
DEFAULT_SKIP_REASONS = {
    6:  "Soft-NMS patch broken: mAP 0.497→0.206, postprocess 200× (see project_soft_nms_broken)",
    7:  "TTA val_only: 论文已有数据，每次裸跑都会重跑 ~10min",
    12: "YOLOv8s baseline: 外部对比基线，二区目标暂不必跑（论文需要时再开）",
    13: "YOLOv8l baseline: 同上 + 8GB 显卡需 batch=2 才能跑通",
    15: "P5 MCA: 2026-05-22 epoch 32 手动停训，CSAC/MCA-localization 叙事作废",
}
DEFAULT_SKIP = list(DEFAULT_SKIP_REASONS.keys())

EXPERIMENTS_BY_IDX = {e["idx"]: e for e in EXPERIMENTS}

# 论文优化版主线队列。裸跑 `python scripts/train.py` 即按此顺序训练 (imgsz/batch 由上面 HP=1280/2 控制),
# 每个练完自动 test 评估。全部带 _1280[_sN] run 后缀 → 写入新目录, 不覆盖旧 960 结果 (留作分辨率消融)。
# (idx, seed, suffix)。越靠前越关键, 可随时 Ctrl+C, 再跑自动续训。
PAPER_QUEUE = [
    # ===== 双骨干泛化主线 (2026-06-21 重排) =====
    # 论文主线 = scale-aware LA + MCA 骨干无关: v8(E01→E16', 已 3-seed) + v11(E18→E24).
    # 裸跑 `python scripts/train.py` 一次点击即: ①秒跳过已完成项 (test/buckets 已在盘) →
    # ②续训 E24 到 150 (auto test+bucket eval) → ③训练 E18-batch1 公平基线 (auto test+bucket eval).
    # 全程可 Ctrl+C, 再点 Run 自动续训.
    #
    # --- 已完成 (训练 + test + size-buckets 均已在盘): 裸跑秒跳过, auto_eval_test 幂等跳过; 留作 provenance ---
    (17, 42, "_1280"),      #   E16' v8 hero seed42 (主)
    (17, 1,  "_1280_s1"),   #   E16' v8 hero seed1
    (17, 7,  "_1280_s7"),   #   E16' v8 hero seed7
    (2,  42, "_1280"),      #   +P2  1280 诊断链中间行
    (1,  42, "_1280"),      #   baseline 1280 诊断链锚点
    (19, 42, "_1280"),      #   E18 v11m baseline (batch=2; 留作 batch 敏感性参考)
    # --- 剩余训练 (这次点击真正要跑完的部分) ---
    (24, 42, "_1280"),      # ★ RESUME E24 108→150 (2026-06-25 GO 达成→补满至与 E25 同 150ep 对称; best.pt 现 ep99 已平台期, 补完须确认 best 是否变, 变了则删 eval_test[_buckets] 重评). YOLOv11m+P2+MCA+scale-aware(HI=32) batch=1
    (25, 42, "_1280"),      # ★ E18 batch=1 公平基线 (apples-to-apples vs E24)
    # --- Phase 2 显著性: 2026-06-25 GO 达成 (E24 在 test mAP@0.5:0.95=0.2861>0.2838 且 mAP_small=0.2490>0.2411 双双赢过公平基线 E25) → 解锁 3-seed 坐实薄胜 mean±std ---
    # --- 先把 E24 (v11 hero) 跑完: seed1 → seed7, 凑齐 3-seed ---
    (24, 1,  "_1280_s1"),   #   E24 seed=1  (当前在训)
    (24, 7,  "_1280_s7"),   #   E24 seed=7  (E24 收尾)
    # --- ① 真 10 类 VisDrone 官方 val 锚点 (2026-07-04 修正): 旧 _vd10_960 实际跑成三类,
    #     因 ensure_data_yaml 未区分数据集且 NMV_DATA 未传给子进程。这里改用真 D:/visdrone10_yolo,
    #     run 名改 _vis10_960, 官方 val split, val large≈1068 个目标, 能支撑公开基准校准。---
    (1,  42, "_vis10_960", {"data": "visdrone10.yaml", "imgsz": 960, "eval_split": "val", "eval_imgsz": 960}),   # true VisDrone10 baseline
    (2,  42, "_vis10_960", {"data": "visdrone10.yaml", "imgsz": 960, "eval_split": "val", "eval_imgsz": 960}),   # true VisDrone10 +P2
    (17, 42, "_vis10_960", {"data": "visdrone10.yaml", "imgsz": 960, "eval_split": "val", "eval_imgsz": 960}),   # true VisDrone10 +P2+MCA+SA
    # --- ② 继续 v8 headline (E01/E02) 补满 3 seed, 使 +0.58/+0.67pp 能做配对 t 检验 + 95%CI。
    #     注: 整体增益落在 ±std 带内、t 检验大概率 *不显著*; 跑它是为「诚实地报一个检验」, 显著性看大目标桶/诊断 Δ ---
    (1,  1,  "_1280_s1"),   #   E01 baseline seed=1  (已有 checkpoint; 10 类锚点后续训)
    (1,  7,  "_1280_s7"),   #   E01 baseline seed=7
    (2,  1,  "_1280_s1"),   #   E02 +P2 seed=1       (−Δ 机理显著性)
    (2,  7,  "_1280_s7"),   #   E02 +P2 seed=7
    # dim/depth 消融 E20-23 不进此 1280 队列; 必要时 960 单 seed 另跑.
]


def apply_run_suffix(exp):
    """If NMV_RUN_SUFFIX is set in env, return a copy of exp with the suffix
    appended to the run name. Used by run_multi_seed.py to keep seed runs
    isolated (e.g. E08_p2_mca_s1 vs E08_p2_mca)."""
    suffix = os.environ.get("NMV_RUN_SUFFIX", "")
    if not suffix:
        return exp
    out = dict(exp)
    out["name"] = exp["name"] + suffix
    return out


def banner(text, ch="="):
    print()
    print(ch * 78)
    print(f"  {text}")
    print(ch * 78)


def fmt_metrics(m):
    if not m:
        return "  (no metrics — training did not complete)"
    map_key = next((k for k in m if "mAP50" in k and "95" not in k), None)
    map95_key = next((k for k in m if "mAP50-95" in k or "mAP50_95" in k), None)
    p_key = next((k for k in m if k.startswith("metrics/precision")), None)
    r_key = next((k for k in m if k.startswith("metrics/recall")), None)
    parts = [f"epoch={m.get('epoch', '?')}"]
    if p_key:
        parts.append(f"P={m[p_key]:.4f}")
    if r_key:
        parts.append(f"R={m[r_key]:.4f}")
    if map_key:
        parts.append(f"mAP50={m[map_key]:.4f}")
    if map95_key:
        parts.append(f"mAP50-95={m[map95_key]:.4f}")
    return "  " + "  ".join(parts)


def run_experiment_in_process(exp):
    """Called inside the subprocess. Builds + trains/vals one experiment."""
    from ultralytics import YOLO

    exp = apply_run_suffix(exp)
    run_dir = RUNS / exp["name"]
    last_pt = run_dir / "weights" / "last.pt"
    hp = dict(HP)
    if os.environ.get("NMV_EPOCHS"):
        hp["epochs"] = int(os.environ["NMV_EPOCHS"])
        print(f"  [NMV_EPOCHS override] epochs={hp['epochs']}")
    if os.environ.get("NMV_WORKERS"):
        hp["workers"] = int(os.environ["NMV_WORKERS"])
        print(f"  [NMV_WORKERS override] workers={hp['workers']}")
    if os.environ.get("NMV_IMGSZ"):
        hp["imgsz"] = int(os.environ["NMV_IMGSZ"])
        print(f"  [NMV_IMGSZ override] imgsz={hp['imgsz']}")
    if os.environ.get("NMV_CACHE"):
        hp["cache"] = os.environ["NMV_CACHE"]
        print(f"  [NMV_CACHE override] cache={hp['cache']}")
    if os.environ.get("NMV_PATIENCE"):
        hp["patience"] = int(os.environ["NMV_PATIENCE"])
        print(f"  [NMV_PATIENCE override] patience={hp['patience']}")
    if os.environ.get("NMV_BATCH"):
        hp["batch"] = int(os.environ["NMV_BATCH"])
        print(f"  [NMV_BATCH override] batch={hp['batch']}")
    if os.environ.get("NMV_SEED"):
        hp["seed"] = int(os.environ["NMV_SEED"])
        print(f"  [NMV_SEED override] seed={hp['seed']}")

    if exp["kind"] == "train":
        if last_pt.exists() and not is_completed(run_dir):
            print(f"  Checkpoint found, resuming from {last_pt}")
            print(f"  resume overrides: workers={hp['workers']}, cache={hp['cache']!r}")
            model = YOLO(str(last_pt))
            model.train(resume=True, workers=hp["workers"], cache=hp["cache"])
        elif exp["cfg"]:
            model = YOLO(exp["cfg"]).load(str(WEIGHTS / exp["weights"]))
            model.train(
                data=str(DATA),
                project=str(RUNS),
                name=exp["name"],
                exist_ok=True,
                **hp,
            )
        else:
            model = YOLO(str(WEIGHTS / exp["weights"]))
            model.train(
                data=str(DATA),
                project=str(RUNS),
                name=exp["name"],
                exist_ok=True,
                **hp,
            )
    elif exp["kind"] == "val_only":
        src_run = RUNS / exp["source"] / "weights" / "best.pt"
        if not src_run.exists():
            print(f"[ERROR] Source weights not found: {src_run}")
            print("        Run the producing experiment first (e.g., --only 4 for E04).")
            sys.exit(2)
        model = YOLO(str(src_run))
        val_kwargs = dict(
            data=str(DATA),
            imgsz=hp["imgsz"],
            batch=hp["batch"],
            workers=0,
            project=str(RUNS),
            name=exp["name"],
            exist_ok=True,
            verbose=True,
            split="val",
        )
        val_kwargs.update(exp.get("val_kwargs", {}))
        model.val(**val_kwargs)
    else:
        raise ValueError(f"Unknown experiment kind: {exp['kind']}")


def run_with_watchdog(cmd, env, exp, run_dir):
    """subprocess.Popen + 心跳轮询的看门狗 (2026-06-10 加, 修两次整夜 hang 损失).

    hang 的形态是子进程不退出但 GPU util ~2%、results.csv 数小时不涨,
    阻塞的 subprocess.run 等不到退出码 → 既有 crash-retry 永不触发。
    这里以 results.csv / last.pt 的 mtime 为心跳, 停滞超阈值就杀进程返回
    伪 returncode, 让上层按 crash 走既有 resume 路径 (从 last.pt 续, epoch 不丢)。

    - train kind: 首心跳宽限 NMV_WATCHDOG_FIRST_MIN (默认 120min, 建 disk cache +
      首 epoch 慢); 其后停滞 > NMV_WATCHDOG_MIN (默认 90min, 最慢 26min/epoch 的
      3 倍余量) 判 hang。
    - val_only kind: 无 results.csv 心跳, 用总时长硬超时 NMV_WATCHDOG_MIN。
    """
    POLL_SEC = 60
    # epoch @1280/batch1 ~20-23min; 停滞阈值取 ~2 个 epoch 余量。2026-06-23 由 90/120
    # 下调到 45/60 (默认), 并修复 resume 误套首心跳宽限的 bug (见下面分支)。
    stale_limit = float(os.environ.get("NMV_WATCHDOG_MIN", "45")) * 60
    first_limit = float(os.environ.get("NMV_WATCHDOG_FIRST_MIN", "60")) * 60
    heartbeat_files = [run_dir / "results.csv", run_dir / "weights" / "last.pt"]

    def latest_heartbeat():
        ts = [f.stat().st_mtime for f in heartbeat_files if f.exists()]
        return max(ts) if ts else None

    proc = subprocess.Popen(cmd, env=env)
    start = time.time()
    while True:
        rc = proc.poll()
        if rc is not None:
            return rc
        time.sleep(POLL_SEC)
        now = time.time()
        if exp["kind"] == "train":
            hb = latest_heartbeat()
            if hb is None:
                # 全新启动, 盘上还没有任何 checkpoint: 首 epoch + disk-cache 构建可能偏慢
                stalled, limit = now - start, first_limit
            elif hb < start:
                # resume: checkpoint 来自上一次, 本次启动后尚无新心跳。模型已热,
                # 续训首 epoch ≈ 常规 epoch → 用常规停滞阈值, 不再误套首心跳宽限。
                # (旧 bug: 每次 resume 都重置成 120min 宽限, 续上又卡时迟迟不杀)
                stalled, limit = now - start, stale_limit
            else:
                stalled, limit = now - hb, stale_limit
        else:
            stalled, limit = now - start, stale_limit
        if stalled > limit:
            print(f"[WATCHDOG] {exp['name']} no heartbeat for {stalled/60:.0f} min "
                  f"(limit {limit/60:.0f} min) — killing pid {proc.pid}, "
                  f"will go through normal crash-retry/resume path")
            proc.kill()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)])
            return -888  # 伪 returncode, 上层按 crash 处理


def _epochs_logged(run_dir):
    """已写入 results.csv 的 epoch 数 (进度信号; 跨 resume 单调递增)。读失败返回 0。"""
    f = run_dir / "results.csv"
    if not f.exists():
        return 0
    try:
        with open(f, encoding="utf-8") as fh:
            return max(0, sum(1 for line in fh if line.strip()) - 1)  # 去表头
    except Exception:
        return 0


def dispatch_one(exp, force=False):
    """Run one experiment in a subprocess with crash-resume retry.

    Returns metrics dict on success. 进度感知重试: 只要每轮相比上次有"新 epoch 落盘"
    就重置失败预算; 仅当连续 MAX_ATTEMPTS 次"零 epoch 进展"才 sys.exit(1) (这样频繁
    但可恢复的 hang——每跑几个 epoch 卡一次——会一路续训跑完, 不会误停整条流水线)。
    Resume relies on early_stop_resume.py preserving optimizer and epoch state in last.pt.
    """
    exp = apply_run_suffix(exp)
    run_dir = RUNS / exp["name"]
    last_pt = run_dir / "weights" / "last.pt"

    if exp["kind"] == "train" and not force:
        if is_completed(run_dir):
            print(f"[SKIP] {exp['name']} already completed.")
            return read_best(run_dir / "results.csv")
        if last_pt.exists():
            print(f"[RESUME] {exp['name']} — checkpoint found, will resume")

    env = os.environ.copy()
    # Windows 不支持 expandable_segments
    # max_split_size_mb:64 → 比 128 更小，减少 reserved 块"占而不用"
    # garbage_collection_threshold:0.6 → Windows 共享 GPU 内存把"上限"虚增到 ~16GB，
    #   0.8 触发太晚（已 OOM）；0.6 提前到 ~9.6G
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:64,garbage_collection_threshold:0.6",
    )
    for k, v in exp.get("env", {}).items():
        env[k] = v
        print(f"  env: {k}={v}")

    cmd = [sys.executable, str(__file__), "--exp-idx", str(exp["idx"])]

    attempt = 0
    fails_without_progress = 0
    epochs_before = _epochs_logged(run_dir)   # 进入时已落盘的 epoch 数 (resume 时 >0)
    while True:
        attempt += 1
        if attempt > 1:
            banner(f"[{exp['idx']}] {exp['name']} — RETRY (attempt {attempt}, "
                   f"连续无进展 {fails_without_progress}/{MAX_ATTEMPTS}, resume from last.pt)", "!")
            if exp["kind"] == "train" and not last_pt.exists():
                print(f"[FATAL] No last.pt at {last_pt}, cannot resume. Aborting.")
                sys.exit(1)
            # retry 直接走 workers=0：不 spawn 子进程，避开 spawn 期 import torch
            # 申请虚拟内存触发 WinError 1455 / numpy ArrayMemoryError 的两类失败
            env["NMV_WORKERS"] = "0"
            print(f"  retry override: NMV_WORKERS=0 (no DataLoader spawn — saves ~2GB virtual memory)")
            # 给 Windows 时间回收前一次崩溃子进程的 working set + standby memory + page file commit
            print(f"  sleeping 30s for OS to reclaim memory from crashed subprocess...")
            time.sleep(30)

        print(f"  cmd: {' '.join(cmd)}  (attempt {attempt})")
        t0 = datetime.now()
        rc = run_with_watchdog(cmd, env, exp, run_dir)
        elapsed = datetime.now() - t0
        print(f"  -> exit code {rc}, elapsed {elapsed}")

        if rc == 0:
            break

        # rc!=0 但 run_dir 已 completed（极少数下 ultralytics 自然 early-stop 后偶发非 0
        # 退出码），按成功处理
        if exp["kind"] == "train" and is_completed(run_dir):
            print(f"  [INFO] rc={rc} but run_dir is_completed — treating as success")
            break

        # 进度感知重试预算: 只要本轮相比上次有新 epoch 落盘, 就视为"在推进"并重置失败计数,
        # 这样每跑几个 epoch 卡一次的频繁 hang 会一路续训跑完; 仅连续 MAX_ATTEMPTS 次
        # "零 epoch 进展"才判真卡死并 sys.exit (不再用累计 attempt 数, 避免误停整条流水线)。
        epochs_now = _epochs_logged(run_dir)
        if epochs_now > epochs_before:
            print(f"[PROGRESS] {exp['name']} rc={rc} 但已推进 {epochs_before}->{epochs_now} "
                  f"epoch — 重置无进展失败计数 (was {fails_without_progress})")
            fails_without_progress = 0
            epochs_before = epochs_now
        else:
            fails_without_progress += 1
            print(f"[CRASH] {exp['name']} attempt {attempt} rc={rc}, 零进展 (epochs 停在 "
                  f"{epochs_now}); 连续无进展 {fails_without_progress}/{MAX_ATTEMPTS}")
            if fails_without_progress >= MAX_ATTEMPTS:
                print(f"[FATAL] {exp['name']} 连续 {MAX_ATTEMPTS} 次无任何 epoch 进展, 判定真卡死。Stopping pipeline.")
                print(f"        Inspect {run_dir} and re-run with: "
                      f"python scripts/train.py --only {exp['idx']}")
                sys.exit(1)

    if exp["kind"] == "train":
        return read_best(run_dir / "results.csv")
    return None


def auto_eval_test(name, data_yaml=None, split="test", imgsz=1280, batch=2):
    """主线每个模型练完(或已完成被跳过)后, 自动在 test 集评估 @imgsz=1280:
      (1) 标准 mAP (整体 mAP@0.5 / mAP@0.5:0.95 / per-class) — 经 val.py
      (2) COCO 小目标分桶 (mAP_small/medium/large) — 经 eval_size_buckets.py
    两项各自幂等 (已有结果则跳过), 续跑很快; 分桶非致命 (失败不打断训练队列)。
    注意: (1) 已存在时不能提前 return, 否则会漏掉 (2) 对已完成模型的分桶回填。"""
    bp = RUNS / name / "weights" / "best.pt"
    if not bp.exists():
        print(f"  [skip eval] best.pt 不存在 (训练未完成?): {bp}")
        return

    # (1) 标准 test 评估
    data_yaml = data_yaml or (ROOT / "configs" / "data" / "nmv_visdrone_3cls.yaml")
    out = RUNS / f"{name}_eval_{split}"
    if (out / "results.csv").exists():
        print(f"  [skip eval] {out.name} 已存在")
    else:
        print(f"  --- {split} 评估 {name} @imgsz={imgsz} ---")
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "val.py"),
             "--run", name, "--data", str(data_yaml),
             "--split", split, "--imgsz", str(imgsz), "--batch", str(batch)]
        ).returncode
        print(f"  -> eval rc={rc}")

    # (2) COCO 小目标分桶 (论文核心卖点 mAP_small 的关键数字) — 幂等 + 非致命
    bdir = RUNS / f"{name}_eval_{split}_buckets"
    if (bdir / "metrics.json").exists():
        print(f"  [skip buckets] {bdir.name} 已存在")
        return
    print(f"  --- 小目标分桶评估 {name} @imgsz={imgsz} split={split} ---")
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "eval_size_buckets.py"),
             "--run", name, "--data", str(data_yaml),
             "--split", split, "--imgsz", str(imgsz)],
            check=False,
        )
    except Exception as e:
        print(f"  [warn] bucket eval 失败(非致命): {e}")


def _unpack_queue_entry(entry):
    """PAPER_QUEUE 项支持 3 元组 (idx, seed, suffix) 或 4 元组
    (idx, seed, suffix, override)，override 可含:
      {'data': 'xxx.yaml', 'imgsz': 960, 'eval_split': 'val', 'eval_imgsz': 960}
    """
    idx, seed, suffix = entry[:3]
    ov = entry[3] if len(entry) > 3 else {}
    return idx, seed, suffix, ov


def run_paper_queue(force=False):
    """裸跑入口: 训练论文优化版 1280 主线 + 自动 test 评估。"""
    global DATA
    banner("论文优化版主线 (imgsz=1280, batch=2) — 训练 + test 评估", "#")
    base_data = DATA
    base_data_env = os.environ.get("NMV_DATA")
    base_imgsz_env = os.environ.get("NMV_IMGSZ")
    print("  队列 (每个约 56h 实测@1280/b2; 旧结果不受影响, 全部写入带后缀的新目录):")
    for entry in PAPER_QUEUE:
        idx, seed, suffix, ov = _unpack_queue_entry(entry)
        tag = f"  [{ov['data']}@{ov.get('imgsz', '-')}]" if ov.get("data") else ""
        print(f"    {EXPERIMENTS_BY_IDX[idx]['name']}{suffix:12s}  seed={seed}{tag}")
    print("  可随时 Ctrl+C; 再跑 `python scripts/train.py` 自动从断点续训。\n")

    results = []
    t0 = datetime.now()
    for entry in PAPER_QUEUE:
        idx, seed, suffix, ov = _unpack_queue_entry(entry)
        # --- 按项的数据集 / imgsz 覆盖; 无覆盖则还原默认, 保证隔离 ---
        if ov.get("data"):
            DATA = ensure_data_yaml(Path(ROOT / "configs" / "data" / ov["data"]))
            os.environ["NMV_DATA"] = str(DATA)
        else:
            DATA = base_data
            if base_data_env is not None:
                os.environ["NMV_DATA"] = base_data_env
            else:
                os.environ.pop("NMV_DATA", None)
        if ov.get("imgsz"):
            os.environ["NMV_IMGSZ"] = str(ov["imgsz"])
        elif base_imgsz_env is not None:
            os.environ["NMV_IMGSZ"] = base_imgsz_env
        else:
            os.environ.pop("NMV_IMGSZ", None)
        os.environ["NMV_RUN_SUFFIX"] = suffix
        if seed != 42:
            os.environ["NMV_SEED"] = str(seed)
        else:
            os.environ.pop("NMV_SEED", None)
        exp = EXPERIMENTS_BY_IDX[idx]
        name = exp["name"] + suffix
        banner(f"{name}  (seed={seed}) — {exp['desc']}", "#")
        m = dispatch_one(exp, force=force)
        results.append((name, m))
        print(fmt_metrics(m))
        eval_split = ov.get("eval_split", "test")
        eval_imgsz = int(ov.get("eval_imgsz", ov.get("imgsz", 1280)))
        eval_batch = int(ov.get("eval_batch", 2))
        auto_eval_test(name, data_yaml=DATA, split=eval_split, imgsz=eval_imgsz, batch=eval_batch)

    DATA = base_data  # 还原全局, 避免污染后续调用
    if base_data_env is not None:
        os.environ["NMV_DATA"] = base_data_env
    else:
        os.environ.pop("NMV_DATA", None)

    banner(f"主线训练完成 — 用时 {datetime.now() - t0}", "#")
    print(f"  {'name':<28}{'mAP50':>10}{'mAP50-95':>12}")
    print("  " + "-" * 50)
    for name, m in results:
        if not m:
            print(f"  {name:<28}  (no metrics)"); continue
        mk = next((k for k in m if "mAP50" in k and "95" not in k), None)
        m9 = next((k for k in m if "mAP50-95" in k or "mAP50_95" in k), None)
        f = lambda k: f"{m[k]:.4f}" if k and k in m else "-"
        print(f"  {name:<28}{f(mk):>10}{f(m9):>12}")
    print("\n  下一步 (可选, 免训增益与论文素材, 各自一条命令):")
    print("    python scripts/sahi_predict.py --run E08_p2_mca_1280 --split test")
    print("    python scripts/sahi_predict.py --run E01_baseline_1280 --split test")
    print("    python scripts/aggregate_seeds.py     # 3 seed mean±std + 显著性")
    print("    python scripts/export_table.py        # 主对比表 (含 960-vs-1280)")
    print(f"\n  输出根目录: {RUNS}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=int, default=None,
                   help="Only run the experiment with this idx (1-26)")
    p.add_argument("--skip", type=int, nargs="+", default=[],
                   help="Skip these experiment idxs (与 DEFAULT_SKIP 叠加)")
    p.add_argument("--start", type=int, default=0,
                   help="Start from this idx (skip earlier)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if already completed")
    p.add_argument("--include-default-skip", action="store_true",
                   help="包含 DEFAULT_SKIP 中默认跳过的实验（除非用 --only，否则 DEFAULT_SKIP 生效）")
    p.add_argument("--exp-idx", type=int, default=None,
                   help=argparse.SUPPRESS)  # internal: subprocess entry point
    args = p.parse_args()

    if args.exp_idx is not None:
        exp = next(e for e in EXPERIMENTS if e["idx"] == args.exp_idx)
        run_experiment_in_process(exp)
        return

    # ★ 裸跑 (无 --only / --start / --skip / --include-default-skip) = 论文优化版 1280 主线。
    #   想跑旧 EXPERIMENTS 列表请显式带 --only/--start/--skip。
    if (args.only is None and args.start == 0 and not args.skip
            and not args.include_default_skip):
        run_paper_queue(force=args.force)
        return

    # --only 会绕过 DEFAULT_SKIP（用户明确指定哪个 idx，按用户意图执行）
    apply_default_skip = (
        args.only is None and not args.include_default_skip
    )
    if apply_default_skip:
        skipped_by_default = [i for i in DEFAULT_SKIP if i not in args.skip]
        if skipped_by_default:
            print("Default-skipped (use --only <idx> or --include-default-skip to force):")
            for i in skipped_by_default:
                print(f"  [{i}] {DEFAULT_SKIP_REASONS[i]}")

    todo = [
        e for e in EXPERIMENTS
        if (args.only is None or e["idx"] == args.only)
        and e["idx"] >= args.start
        and e["idx"] not in args.skip
        and (not apply_default_skip or e["idx"] not in DEFAULT_SKIP)
    ]
    if not todo:
        print("No experiments to run.")
        return

    banner(f"Non-motor vehicle SOD experiments — {len(todo)} to run", "#")
    for e in todo:
        print(f"  [{e['idx']}] {e['name']:25s} {e['desc']}")

    results = []
    t0 = datetime.now()
    for exp in todo:
        banner(f"[{exp['idx']}] {exp['name']} — {exp['desc']}", "#")
        m = dispatch_one(exp, force=args.force)
        results.append((exp, m))
        print(fmt_metrics(m))

    banner(f"Final summary — total elapsed {datetime.now() - t0}", "#")
    print(f"  {'idx':<5}{'name':<25}{'mAP50':>10}{'mAP50-95':>12}{'P':>10}{'R':>10}")
    print("  " + "-" * 72)
    for exp, m in results:
        if m is None:
            print(f"  {exp['idx']:<5}{exp['name']:<25}  (no metrics)")
            continue
        map_key = next((k for k in m if "mAP50" in k and "95" not in k), None)
        map95_key = next((k for k in m if "mAP50-95" in k or "mAP50_95" in k), None)
        p_key = next((k for k in m if k.startswith("metrics/precision")), None)
        r_key = next((k for k in m if k.startswith("metrics/recall")), None)
        def fmt(k):
            return f"{m[k]:.4f}" if k and k in m else "-"
        print(f"  {exp['idx']:<5}{exp['name']:<25}{fmt(map_key):>10}{fmt(map95_key):>12}{fmt(p_key):>10}{fmt(r_key):>10}")
    print()
    print(f"  Outputs: {RUNS}")


if __name__ == "__main__":
    main()
