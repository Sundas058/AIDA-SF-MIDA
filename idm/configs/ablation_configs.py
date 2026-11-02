"""Controlled module switches; independent of the main multi-run result tables."""
from copy import deepcopy
ABLATION_CONFIGS = {
    "source_baseline": {"use_ms_idg": False, "use_pmr": False, "use_dfc": False},
    "ms_idg": {"use_ms_idg": True, "use_pmr": False, "use_dfc": False},
    "ms_idg_identity": {"use_ms_idg": True, "use_pmr": True, "use_dfc": False, "pmr_relational": False},
    "ms_idg_relational": {"use_ms_idg": True, "use_pmr": True, "use_dfc": False, "pmr_identity": False},
    "ms_idg_pmr": {"use_ms_idg": True, "use_pmr": True, "use_dfc": False},
    "aida_full": {"use_ms_idg": True, "use_pmr": True, "use_dfc": True},
}
def get_ablation_config(name):
    if name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation: {name}")
    return deepcopy(ABLATION_CONFIGS[name])
def list_ablation_configs():
    return sorted(ABLATION_CONFIGS)
def get_ablation_description(name):
    return name.replace("_", " ")
def apply_ablation_to_args(args, name):
    for key, value in get_ablation_config(name).items():
        setattr(args, key, value)
    return args, get_ablation_description(name)

