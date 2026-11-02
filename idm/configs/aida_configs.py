"""Experiment presets aligned with Section 4.1.2 and Algorithm 1."""
from copy import deepcopy

AIDA_DEFAULTS = {
    "arch": "resnet50", "height": 256, "width": 128, "batch_size": 64,
    "num_instances": 4, "workers": 4, "iters": 400, "seed": 1,
    "epochs": 50, "supervised_epochs": 10, "lr": 3.5e-4, "weight_decay": 5e-4,
    "margin": 0.3, "lambda_tri": 1.0, "lambda_rel": 1.0,
    "use_ms_idg": True, "use_pmr": True, "use_dfc": True,
    "pmr_identity": True, "pmr_relational": True,
    "lambda_idg_0": 1.0, "lambda_pmr_0": 1.0,
    "beta_dfc": 0.1, "gamma_dfc": 0.5, "lambda_max": 10.0,
    "entropy_max": 10.0, "grad_window": 100, "scale_max": 1.5,
    "use_gem": True, "use_sam": True, "sam_rho": 0.05,
    "pretrained": True, "num_parts": 4, "memory_slots": 32,
    "random_erasing": 0.5, "eps": 0.6, "min_samples": 4, "k1": 30, "k2": 6,
    "mu1": 0.7, "mu2": 0.1, "mu3": 1.0,
    "data_dir": "data", "logs_dir": None, "device": "auto",
}

def _preset(sources, target, **changes):
    return {**deepcopy(AIDA_DEFAULTS), "sources": sources, "target": target, **changes}

DG_CONFIGS = {
    "market_duke_to_msmt": _preset(["market1501", "dukemtmc"], "msmt17"),
    "market_msmt_to_duke": _preset(["market1501", "msmt17"], "dukemtmc"),
    "duke_msmt_to_market": _preset(["dukemtmc", "msmt17"], "market1501"),
    "market_duke_msmt_to_cuhk03": _preset(["market1501", "dukemtmc", "msmt17"], "cuhk03"),
    "market_duke_msmt_to_personx": _preset(["market1501", "dukemtmc", "msmt17"], "personx"),
    "personx_to_market": _preset(["personx"], "market1501"),
    "personx_to_duke": _preset(["personx"], "dukemtmc"),
    "personx_to_msmt": _preset(["personx"], "msmt17"),
}
SF_PRETRAIN_CONFIGS = {
    "pretrain_for_market": _preset(["dukemtmc", "msmt17"], "market1501"),
    "pretrain_for_duke": _preset(["market1501", "msmt17"], "dukemtmc"),
    "pretrain_for_msmt": _preset(["market1501", "dukemtmc"], "msmt17"),
}
SF_ADAPT_CONFIGS = {
    "adapt_to_market": _preset([], "market1501", epochs=30, supervised_epochs=0, lr=1e-4),
    "adapt_to_duke": _preset([], "dukemtmc", epochs=30, supervised_epochs=0, lr=1e-4),
    "adapt_to_msmt": _preset([], "msmt17", epochs=30, supervised_epochs=0, lr=1e-4),
}
UDA_CONFIGS = {
    f"{source}_to_{target}": _preset([source], target, epochs=60, use_gem=False,
        use_sam=False, num_parts=0, memory_slots=0)
    for source in ("market1501", "dukemtmc", "msmt17")
    for target in ("market1501", "dukemtmc", "msmt17") if source != target
}
GROUPS = {"dg": DG_CONFIGS, "sf_pretrain": SF_PRETRAIN_CONFIGS,
          "sf_adapt": SF_ADAPT_CONFIGS, "uda": UDA_CONFIGS}

def get_config(config_name, config_type="dg"):
    if config_type not in GROUPS or config_name not in GROUPS[config_type]:
        raise ValueError(f"Unknown configuration {config_type}/{config_name}")
    return deepcopy(GROUPS[config_type][config_name])

def list_configs(config_type="dg"):
    if config_type not in GROUPS:
        raise ValueError(f"Unknown configuration group: {config_type}")
    return sorted(GROUPS[config_type])

