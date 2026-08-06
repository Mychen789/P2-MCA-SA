import inspect


_DONE = False


def register():
    """Inject EMA / SimAM into ultralytics.nn.tasks namespace AND patch parse_model
    so they're treated as base_modules (auto-resolves c1/c2 from prev layer).

    Why textual patch: ultralytics' parse_model holds base_modules as a local
    frozenset, so we can't extend it from outside. We re-`exec` the function
    source in the tasks module's namespace with EMA/SimAM appended to the set.
    """
    global _DONE
    if _DONE:
        return

    import ultralytics.nn.tasks as T
    from nmv.modules.attention import EMA, SimAM
    from nmv.modules.mca import MCA
    from nmv.modules.aspp import ASPPLite
    from nmv.modules.transformer import P5Transformer

    setattr(T, "EMA", EMA)
    setattr(T, "SimAM", SimAM)
    setattr(T, "MCA", MCA)
    setattr(T, "ASPPLite", ASPPLite)
    setattr(T, "P5Transformer", P5Transformer)

    src = inspect.getsource(T.parse_model)
    anchor = "base_modules = frozenset(\n        {\n            Classify,"
    inject = "base_modules = frozenset(\n        {\n            EMA,\n            SimAM,\n            MCA,\n            ASPPLite,\n            P5Transformer,\n            Classify,"
    if anchor not in src:
        raise RuntimeError(
            "register_modules: parse_model anchor not found — ultralytics version may have changed.\n"
            f"Expected to find:\n{anchor}"
        )
    new_src = src.replace(anchor, inject)
    exec(new_src, vars(T))

    _DONE = True
