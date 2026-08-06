import os

_APPLIED = False


def apply_all():
    global _APPLIED
    if _APPLIED:
        return
    from nmv.patches.register_modules import register
    from nmv.patches.early_stop_resume import install as install_early_stop_resume
    from nmv.patches.oom_defense import install as install_oom_defense
    from nmv.patches.tal_oom_fix import install as install_tal_oom_fix
    from nmv.patches.mpdiou_loss import install as install_mpdiou
    from nmv.patches.soft_nms import install as install_soft_nms
    from nmv.patches.scale_aware_assign import install as install_scale_assign

    register()
    install_early_stop_resume()
    install_oom_defense()
    install_tal_oom_fix()
    install_mpdiou(enable=os.environ.get("NMV_IOU", "ciou").lower() == "mpdiou")
    install_soft_nms(enable=os.environ.get("NMV_NMS", "hard").lower() == "soft")
    install_scale_assign(
        enable=os.environ.get("NMV_SCALE_ASSIGN", "0") == "1",
        audit=os.environ.get("NMV_TAL_AUDIT", "0") == "1",
    )
    _APPLIED = True
