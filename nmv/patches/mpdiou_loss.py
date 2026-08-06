import torch


_ORIG_FWD = None


def _mpdiou_per_box(pred, target, diag_sq, eps=1e-7):
    """MPDIoU as defined in Siliang & Yong, arXiv:2307.07662.

    pred, target: (..., 4) in xyxy
    diag_sq: scalar tensor, image diagonal² for normalization
    Returns IoU-like score; loss is (1 - score).
    """
    inter_x1 = torch.max(pred[..., 0], target[..., 0])
    inter_y1 = torch.max(pred[..., 1], target[..., 1])
    inter_x2 = torch.min(pred[..., 2], target[..., 2])
    inter_y2 = torch.min(pred[..., 3], target[..., 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area_p = (pred[..., 2] - pred[..., 0]).clamp(0) * (pred[..., 3] - pred[..., 1]).clamp(0)
    area_t = (target[..., 2] - target[..., 0]).clamp(0) * (target[..., 3] - target[..., 1]).clamp(0)
    union = area_p + area_t - inter + eps
    iou = inter / union
    d1 = (pred[..., 0] - target[..., 0]).pow(2) + (pred[..., 1] - target[..., 1]).pow(2)
    d2 = (pred[..., 2] - target[..., 2]).pow(2) + (pred[..., 3] - target[..., 3]).pow(2)
    return iou - (d1 + d2) / (diag_sq + eps)


def _mpdiou_forward(
    self,
    pred_dist,
    pred_bboxes,
    anchor_points,
    target_bboxes,
    target_scores,
    target_scores_sum,
    fg_mask,
    imgsz,
    stride,
):
    from ultralytics.utils.tal import bbox2dist
    import torch.nn.functional as F

    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    diag_sq = imgsz[0].pow(2) + imgsz[1].pow(2)
    iou = _mpdiou_per_box(pred_bboxes[fg_mask], target_bboxes[fg_mask], diag_sq).unsqueeze(-1)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = (
            self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        )
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        target_ltrb = bbox2dist(anchor_points, target_bboxes)
        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= imgsz[1]
        pred_dist[..., 1::2] /= imgsz[0]
        loss_dfl = (
            F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
        )
        loss_dfl = loss_dfl.sum() / target_scores_sum

    return loss_iou, loss_dfl


def install(enable=True):
    """Replace BboxLoss.forward with the MPDIoU variant. Idempotent + reversible."""
    global _ORIG_FWD
    from ultralytics.utils.loss import BboxLoss

    if not enable:
        if _ORIG_FWD is not None:
            BboxLoss.forward = _ORIG_FWD
            _ORIG_FWD = None
        return

    if _ORIG_FWD is None:
        _ORIG_FWD = BboxLoss.forward
    BboxLoss.forward = _mpdiou_forward
