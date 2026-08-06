"""Fast CPU unit test for scale_aware_assign patch. Not part of the pipeline.

Verifies that with NMV_SCALE_ASSIGN=1, a large GT is forbidden from low-stride
(high-res) heads and only allowed at the largest stride, while a small GT keeps
positives at the smallest stride.
"""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["NMV_SCALE_ASSIGN"] = "1"
os.environ["NMV_SCALE_HI_RATIO"] = "16.0"
os.environ["NMV_SCALE_LO_RATIO"] = "0.0"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import nmv  # noqa: E402,F401  triggers patch install
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors  # noqa: E402

STRIDES = [4, 8, 16, 32]
IMG = 512

# Build dummy feats so the (patched) make_anchors stashes a real stride_tensor.
feats = [torch.zeros(1, 1, IMG // s, IMG // s) for s in STRIDES]
anchor_points, stride_tensor = make_anchors(feats, STRIDES, 0.5)
anc_points_px = anchor_points * stride_tensor  # pixel coords, as the loss passes
n_anchors = anc_points_px.shape[0]
stride_per_anchor = stride_tensor.squeeze(-1)
print(f"n_anchors={n_anchors}  per-level counts:",
      {int(s): int((stride_per_anchor == s).sum()) for s in STRIDES})

# Two GTs at image center: small (max-side 20) and large (max-side 300).
gt_bboxes = torch.tensor([[[246., 246., 266., 266.],    # small, side 20
                           [106., 106., 406., 406.]]])   # large, side 300
gt_labels = torch.tensor([[[0], [0]]], dtype=torch.float32)
mask_gt = torch.ones(1, 2, 1)

assigner = TaskAlignedAssigner(topk=13, num_classes=3, stride=[float(s) for s in STRIDES])
assigner.bs = 1
assigner.n_max_boxes = 2

# Dummy predictions: a box at each anchor sized ~ 4*stride, constant score.
half = (stride_per_anchor * 2).unsqueeze(-1)
pd_bboxes = torch.cat([anc_points_px - half, anc_points_px + half], dim=-1).unsqueeze(0)
pd_scores = torch.full((1, n_anchors, 3), 0.5)

mask_pos, align_metric, overlaps = assigner.get_pos_mask(
    pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points_px, mask_gt
)

# mask_pos: (b, n_max_boxes, n_anchors)
small_pos = mask_pos[0, 0]   # small GT
large_pos = mask_pos[0, 1]   # large GT


def pos_per_level(pos):
    return {int(s): int(pos[stride_per_anchor == s].sum()) for s in STRIDES}


print("small GT positives per level:", pos_per_level(small_pos))
print("large GT positives per level:", pos_per_level(large_pos))

sp, lp = pos_per_level(small_pos), pos_per_level(large_pos)

ok = True
# Large object (300px) must be forbidden from P2(4)/P3(8)/P4(16); only P5(32).
for s in (4, 8, 16):
    if lp[s] != 0:
        print(f"FAIL: large GT has {lp[s]} positives at stride {s} (expected 0)")
        ok = False
if lp[32] == 0:
    print("FAIL: large GT has 0 positives at stride 32 (expected >0)")
    ok = False
# Small object (20px) must still get positives at the high-res P2 head.
if sp[4] == 0:
    print("FAIL: small GT has 0 positives at stride 4 (expected >0)")
    ok = False

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
