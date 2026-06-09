"""
AIDA Ablation Study Configurations

Predefined combinations of AIDA flags for ablation studies:
1. IDM++ baseline (all AIDA flags False)
2. +MS-IDG
3. +MS-IDG + PMR
4. +MS-IDG + PMR + DFC (full AIDA/SF-MIDA)
"""

from .aida_configs import AIDA_DEFAULTS

# Ablation configurations
ABLATION_CONFIGS = {
    # 1. IDM++ Baseline: All AIDA flags False
    'idmpp_baseline': {
        'use_ms_idg': False,
        'use_pmr': False,
        'use_dfc': False,
        'description': 'IDM++ baseline (all AIDA flags False)'
    },
    
    # 2. +MS-IDG: Only Multi-Source Intermediate Domain Generator
    'idmpp_ms_idg': {
        'use_ms_idg': True,
        'use_pmr': False,
        'use_dfc': False,
        'lambda_idg_0': AIDA_DEFAULTS['lambda_idg_0'],
        'num_sources': AIDA_DEFAULTS['num_sources'],
        'channel': AIDA_DEFAULTS['channel'],
        'description': 'IDM++ + MS-IDG (Multi-Source Intermediate Domain Generator)'
    },
    
    # 3. +MS-IDG + PMR: MS-IDG with Pseudo-Mirror Regularization
    'idmpp_ms_idg_pmr': {
        'use_ms_idg': True,
        'use_pmr': True,
        'use_dfc': False,
        'lambda_idg_0': AIDA_DEFAULTS['lambda_idg_0'],
        'lambda_pmr_0': AIDA_DEFAULTS['lambda_pmr_0'],
        'lambda_sd': AIDA_DEFAULTS['lambda_sd'],
        'num_sources': AIDA_DEFAULTS['num_sources'],
        'channel': AIDA_DEFAULTS['channel'],
        'description': 'IDM++ + MS-IDG + PMR (Pseudo-Mirror Regularization)'
    },
    
    # 4. Full AIDA: MS-IDG + PMR + DFC
    'aida_full': {
        'use_ms_idg': True,
        'use_pmr': True,
        'use_dfc': True,
        **AIDA_DEFAULTS,
        'description': 'Full AIDA/SF-MIDA (MS-IDG + PMR + DFC)'
    },
}


def get_ablation_config(ablation_name):
    """
    Get ablation configuration by name.
    
    Args:
        ablation_name: One of 'idmpp_baseline', 'idmpp_ms_idg', 
                      'idmpp_ms_idg_pmr', 'aida_full'
    
    Returns:
        dict: Configuration dictionary with AIDA flags set
    """
    if ablation_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation config: {ablation_name}. "
                        f"Available: {list(ABLATION_CONFIGS.keys())}")
    return ABLATION_CONFIGS[ablation_name].copy()


def list_ablation_configs():
    """List all available ablation configurations."""
    return list(ABLATION_CONFIGS.keys())


def get_ablation_description(ablation_name):
    """Get human-readable description of an ablation config."""
    config = get_ablation_config(ablation_name)
    return config.get('description', ablation_name)


def apply_ablation_to_args(args, ablation_name):
    """
    Apply ablation configuration to argparse Namespace.
    
    Args:
        args: argparse.Namespace object
        ablation_name: Name of ablation config to apply
    
    Returns:
        argparse.Namespace: Modified args object
    """
    config = get_ablation_config(ablation_name)
    
    # Remove description before applying
    description = config.pop('description', None)
    
    # Apply config values to args
    for key, value in config.items():
        setattr(args, key, value)
    
    return args, description

