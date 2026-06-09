#!/usr/bin/env python
"""
AIDA/SF-MIDA Training Script with Config Support

This script allows running AIDA experiments using predefined configurations
from idm.configs.aida_configs.
"""

from __future__ import print_function, absolute_import
import argparse
import sys
import os.path as osp

# Add parent directory to path
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from idm.configs import get_config, list_configs
from examples.train_idm import main_worker
import argparse as ap


def apply_config_to_args(args, config):
    """Apply configuration dictionary to argparse Namespace."""
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            # Add new attributes
            setattr(args, key, value)
    return args


def main():
    parser = argparse.ArgumentParser(
        description="AIDA/SF-MIDA training with config support"
    )
    
    # Config selection
    parser.add_argument('--config-type', type=str, default='dg',
                        choices=['dg', 'sf_pretrain', 'sf_adapt'],
                        help='Type of configuration: dg (domain generalization), '
                             'sf_pretrain (source-free pre-training), '
                             'sf_adapt (source-free adaptation)')
    parser.add_argument('--config-name', type=str, required=True,
                        help='Name of configuration to use. Use --list-configs to see available configs.')
    parser.add_argument('--list-configs', action='store_true',
                        help='List available configurations and exit')
    
    # Override config with command-line args (optional)
    parser.add_argument('--override', type=str, nargs='+',
                        help='Override config values: key1=value1 key2=value2')
    
    # Parse known args first to check for --list-configs
    args, remaining = parser.parse_known_args()
    
    if args.list_configs:
        print("Available configurations:")
        for config_type in ['dg', 'sf_pretrain', 'sf_adapt']:
            print(f"\n{config_type.upper()}:")
            configs = list_configs(config_type)
            for cfg_name in configs:
                print(f"  - {cfg_name}")
        return
    
    # Get configuration
    try:
        config = get_config(args.config_name, args.config_type)
        if not config:
            print(f"Error: Configuration '{args.config_name}' not found for type '{args.config_type}'")
            print(f"Available configs: {list_configs(args.config_type)}")
            return
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Apply overrides if provided
    if args.override:
        for override in args.override:
            if '=' in override:
                key, value = override.split('=', 1)
                # Try to convert to appropriate type
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                config[key] = value
    
    # Import and use the main training script's argument parser
    # We'll create a minimal args object and populate it from config
    from examples import train_idm
    
    # Create args namespace from config
    class ConfigArgs:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
            # Set defaults for required args not in config
            if not hasattr(self, 'data_dir'):
                self.data_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'data')
            if not hasattr(self, 'logs_dir'):
                self.logs_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'logs')
            if not hasattr(self, 'print_freq'):
                self.print_freq = 50
            if not hasattr(self, 'eval_step'):
                self.eval_step = 1
            if not hasattr(self, 'seed'):
                self.seed = 1
            if not hasattr(self, 'workers'):
                self.workers = 4
            if not hasattr(self, 'batch_size'):
                self.batch_size = 64
            if not hasattr(self, 'height'):
                self.height = 256
            if not hasattr(self, 'width'):
                self.width = 128
            if not hasattr(self, 'num_instances'):
                self.num_instances = 4
            if not hasattr(self, 'iters'):
                self.iters = 400
            if not hasattr(self, 'weight_decay'):
                self.weight_decay = 5e-4
            if not hasattr(self, 'step_size'):
                self.step_size = 20
            if not hasattr(self, 'margin'):
                self.margin = 0.3
            if not hasattr(self, 'features'):
                self.features = 0
            if not hasattr(self, 'dropout'):
                self.dropout = 0.0
            if not hasattr(self, 'memorySize'):
                self.memorySize = 8192
            if not hasattr(self, 'ratio'):
                self.ratio = 1.0
            if not hasattr(self, 'featureSize'):
                self.featureSize = 2048
            if not hasattr(self, 'use_xbm'):
                self.use_xbm = False
            if not hasattr(self, 'eps'):
                self.eps = 0.6
            if not hasattr(self, 'k1'):
                self.k1 = 30
            if not hasattr(self, 'k2'):
                self.k2 = 6
            if not hasattr(self, 'resume'):
                self.resume = ''
            if not hasattr(self, 'source_free'):
                self.source_free = False
    
    args_obj = ConfigArgs(config)
    
    # Set logs directory based on config name
    if not hasattr(args_obj, 'logs_dir') or args_obj.logs_dir == 'logs':
        config_suffix = f"_{args.config_type}_{args.config_name}"
        args_obj.logs_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 
                                               'logs', f'aida{config_suffix}')
    
    print("=" * 60)
    print(f"AIDA Configuration: {args.config_name} ({args.config_type})")
    print("=" * 60)
    print(f"Sources: {getattr(args_obj, 'sources', 'N/A')}")
    print(f"Target: {getattr(args_obj, 'target', 'N/A')}")
    print(f"AIDA Modules: MS-IDG={getattr(args_obj, 'use_ms_idg', False)}, "
          f"PMR={getattr(args_obj, 'use_pmr', False)}, "
          f"DFC={getattr(args_obj, 'use_dfc', False)}")
    print("=" * 60)
    
    # Call main_worker
    main_worker(args_obj)


if __name__ == '__main__':
    main()

