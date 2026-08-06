import time

import torch


_ORIG_NMS = None
_SOFT_SIGMA = 0.5
_SOFT_SCORE_THRESH = 0.001


def _box_iou_xyxy(a, b):
    """Pairwise IoU between two sets of xyxy boxes.

    a: (N, 4), b: (M, 4) -> (N, M)
    """
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def _soft_nms_one_class(boxes, scores, iou_thres, sigma, score_thresh, max_det):
    """Per-class Soft-NMS with Gaussian decay. Returns indices into the input.

    Loops only over kept boxes (typically <= max_det), tensor ops inside.
    """
    keep = []
    idxs = torch.arange(boxes.size(0), device=boxes.device)
    s = scores.clone()
    while idxs.numel() > 0 and len(keep) < max_det:
        m = s.argmax()
        i = idxs[m].item()
        keep.append(i)
        if idxs.numel() == 1:
            break
        # mask out the picked one
        mask = torch.ones_like(idxs, dtype=torch.bool)
        mask[m] = False
        idxs = idxs[mask]
        s = s[mask]
        # decay scores by Gaussian over IoU with the picked box
        ious = _box_iou_xyxy(boxes[i].unsqueeze(0), boxes[idxs])[0]
        decay = torch.exp(-(ious ** 2) / sigma)
        s = s * decay
        # drop boxes whose decayed score fell below threshold
        keep_mask = s > score_thresh
        idxs = idxs[keep_mask]
        s = s[keep_mask]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _soft_nms(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
):
    """Soft-NMS replacement for ultralytics.utils.nms.non_max_suppression.

    Mirrors the original signature; replaces only the per-image, per-class NMS step
    with Gaussian Soft-NMS while preserving all upstream filtering.
    """
    from ultralytics.utils.ops import xywh2xyxy

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    extra = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres

    multi_label &= nc > 1
    prediction = prediction.transpose(-1, -2)
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    t0 = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    time_limit = 2.0 + max_time_img * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        box, cls, mask_extra = x.split((4, nc, extra), 1)
        if multi_label:
            ij = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[ij[0]], cls[ij[0], ij[1], None], ij[1].unsqueeze(-1).float(), mask_extra[ij[0]]), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask_extra), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]
        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes_off = x[:, :4] + c
        keep = _soft_nms_one_class(
            boxes_off, x[:, 4], iou_thres, _SOFT_SIGMA, _SOFT_SCORE_THRESH, max_det
        )
        output[xi] = x[keep]
        if (time.time() - t0) > time_limit:
            break

    return output


def install(enable=True):
    """Replace ultralytics.utils.nms.non_max_suppression with Soft-NMS. Reversible."""
    global _ORIG_NMS
    import ultralytics.utils.nms as nms_mod

    if not enable:
        if _ORIG_NMS is not None:
            nms_mod.non_max_suppression = _ORIG_NMS
            _ORIG_NMS = None
        return

    if _ORIG_NMS is None:
        _ORIG_NMS = nms_mod.non_max_suppression
    nms_mod.non_max_suppression = _soft_nms
