"""
AIDA/SF-MIDA Experimental Configuration Presets

This module provides configuration presets for different experimental setups
as described in the AIDA/SF-MIDA paper.
"""

# Default AIDA hyperparameters (from paper)
AIDA_DEFAULTS = {
    'use_ms_idg': True,
    'use_pmr': True,
    'use_dfc': True,
    'lambda_idg_0': 1.0,
    'lambda_pmr_0': 1.0,
    'lambda_dfc_0': 1.0,
    'lambda_sd': 1.0,
    'beta_dfc': 0.1,
    'lambda_max': 10.0,
    'use_entropy_modulation': True,
    'entropy_weight': 0.5,
    'num_sources': 3,  # For MS-IDG
    'channel': 2048,   # Feature channel size
    # DFC loss hyperparameters
    'entropy_target': 2.0,      # Target entropy value
    'gradvar_target': 0.1,      # Target gradient variance
    'w_entropy': 0.1,           # Weight for entropy term in DFC loss
    'w_grad': 0.1,              # Weight for gradient variance term in DFC loss
    # Model architecture
    'use_gem': True,            # Use GeM pooling for AIDA experiments (matches paper)
    'use_vit_blocks': True,     # Use lightweight ViT blocks on top of CNN features
    'vit_num_blocks': 2,        # Number of ViT blocks (default: 2)
    'vit_num_heads': 4,        # Number of attention heads per block (default: 4)
    'vit_mlp_ratio': 4.0,      # MLP hidden dimension ratio (default: 4.0)
    # Optimizer (for AIDA runs: AdamW + Cosine Annealing + SAM)
    'optimizer': 'adamw',        # Optimizer type: 'adam' or 'adamw' (default: 'adamw' for AIDA)
    'lr_sched': 'cosine',        # Learning rate scheduler: 'step' or 'cosine' (default: 'cosine' for AIDA)
    'use_sam': True,             # Use SAM (Sharpness-Aware Minimization) optimizer
    'sam_rho': 0.05,             # SAM neighborhood size (default: 0.05)
    'sam_adaptive': False,        # Use adaptive rho per parameter (default: False)
}

# Multi-Source Domain Generalization (DG) Configs
# Leave-one-out: train on 2 sources, test on the left-out one

DG_CONFIGS = {
    # Train on Market + Duke, test on MSMT
    'market_duke_to_msmt': {
        'sources': ['market1501', 'dukemtmc'],
        'target': 'msmt17',
        'arch': 'resnet50_idm',
        'stage': 0,
        'mu1': 0.7,
        'mu2': 0.1,
        'mu3': 1.0,
        'use_xbm': True,
        **AIDA_DEFAULTS
    },
    
    # Train on Market + MSMT, test on Duke
    'market_msmt_to_duke': {
        'sources': ['market1501', 'msmt17'],
        'target': 'dukemtmc',
        'arch': 'resnet50_idm',
        'stage': 0,
        'mu1': 0.7,
        'mu2': 0.1,
        'mu3': 1.0,
        'use_xbm': True,
        **AIDA_DEFAULTS
    },
    
    # Train on Duke + MSMT, test on Market
    'duke_msmt_to_market': {
        'sources': ['dukemtmc', 'msmt17'],
        'target': 'market1501',
        'arch': 'resnet50_idm',
        'stage': 0,
        'mu1': 0.7,
        'mu2': 0.1,
        'mu3': 1.0,
        'use_xbm': True,
        **AIDA_DEFAULTS
    },
    
    # Train on all three, test on PersonX (synthetic)
    'market_duke_msmt_to_personx': {
        'sources': ['market1501', 'dukemtmc', 'msmt17'],
        'target': 'personx',
        'arch': 'resnet50_idm',
        'stage': 0,
        'mu1': 0.7,
        'mu2': 0.1,
        'mu3': 1.0,
        'use_xbm': True,
        'num_sources': 3,
        **AIDA_DEFAULTS
    },
}

# Source-Free Multi-Source Intermediate Domain Adaptation (SF-MIDA) Configs

# Pre-training phase: train on multiple sources with AIDA
SF_PRETRAIN_CONFIGS = {
    'pretrain_market_duke_msmt': {
        'sources': ['market1501', 'dukemtmc', 'msmt17'],
        'target': None,  # No target in pre-training
        'arch': 'resnet50_idm',
        'stage': 0,
        'mu1': 0.7,
        'mu2': 0.1,
        'mu3': 1.0,
        'use_xbm': True,
        'num_sources': 3,
        'epochs': 50,  # Pre-training epochs
        **AIDA_DEFAULTS
    },
}

# Adaptation phase: adapt pre-trained model to new target without source images
SF_ADAPT_CONFIGS = {
    'adapt_to_market': {
        'pretrained_checkpoint': None,  # To be set when running
        'target': 'market1501',
        'arch': 'resnet50_idm',
        'stage': 0,
        'mu1': 0.7,
        'mu2': 0.1,
        'mu3': 1.0,
        'use_xbm': True,
        'epochs': 30,  # Adaptation epochs (typically fewer)
        'lr': 0.0001,  # Lower learning rate for adaptation
        # AIDA modules enabled for source-free adaptation
        **AIDA_DEFAULTS
    },
    
    'adapt_to_duke': {
        'pretrained_checkpoint': None,
        'target': 'dukemtmc',
        'arch': 'resnet50_idm',
        'stage': 0,
        'mu1': 0.7,
        'mu2': 0.1,
        'mu3': 1.0,
        'use_xbm': True,
        'epochs': 30,
        'lr': 0.0001,
        **AIDA_DEFAULTS
    },
    
    'adapt_to_msmt': {
        'pretrained_checkpoint': None,
        'target': 'msmt17',
        'arch': 'resnet50_idm',
        'stage': 0,
        'mu1': 0.7,
        'mu2': 0.1,
        'mu3': 1.0,
        'use_xbm': True,
        'epochs': 30,
        'lr': 0.0001,
        **AIDA_DEFAULTS
    },
}


def get_config(config_name, config_type='dg'):
    """
    Get configuration by name.
    
    Args:
        config_name: Name of the config
        config_type: 'dg' for domain generalization, 'sf_pretrain' for source-free pre-training,
                    'sf_adapt' for source-free adaptation
    
    Returns:
        dict: Configuration dictionary
    """
    if config_type == 'dg':
        return DG_CONFIGS.get(config_name, {})
    elif config_type == 'sf_pretrain':
        return SF_PRETRAIN_CONFIGS.get(config_name, {})
    elif config_type == 'sf_adapt':
        return SF_ADAPT_CONFIGS.get(config_name, {})
    else:
        raise ValueError(f"Unknown config_type: {config_type}")


def list_configs(config_type='dg'):
    """List available configuration names."""
    if config_type == 'dg':
        return list(DG_CONFIGS.keys())
    elif config_type == 'sf_pretrain':
        return list(SF_PRETRAIN_CONFIGS.keys())
    elif config_type == 'sf_adapt':
        return list(SF_ADAPT_CONFIGS.keys())
    else:
        raise ValueError(f"Unknown config_type: {config_type}")

