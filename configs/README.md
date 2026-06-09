# AIDA/SF-MIDA Experimental Configurations

This directory contains configuration presets for AIDA (Adaptive Intermediate Domain Adaptation) and SF-MIDA (Source-Free Multi-Source Intermediate Domain Adaptation) experiments.

## Configuration Types

### 1. Multi-Source Domain Generalization (DG)

Leave-one-out evaluation: Train on 2-3 source datasets, test on the left-out target.

**Available Configs:**
- `market_duke_to_msmt`: Train on Market+Duke, test on MSMT
- `market_msmt_to_duke`: Train on Market+MSMT, test on Duke
- `duke_msmt_to_market`: Train on Duke+MSMT, test on Market
- `market_duke_msmt_to_personx`: Train on all three, test on PersonX (synthetic)

### 2. Source-Free Pre-training

Pre-train a model on multiple sources with AIDA modules enabled.

**Available Configs:**
- `pretrain_market_duke_msmt`: Pre-train on Market+Duke+MSMT

### 3. Source-Free Adaptation

Adapt a pre-trained model to a new target dataset without using source images.

**Available Configs:**
- `adapt_to_market`: Adapt to Market-1501
- `adapt_to_duke`: Adapt to DukeMTMC-ReID
- `adapt_to_msmt`: Adapt to MSMT17

## Usage

### Using Shell Scripts

#### Multi-Source Domain Generalization:
```bash
cd scripts
./run_aida_dg.sh market_duke_to_msmt
./run_aida_dg.sh market_msmt_to_duke
./run_aida_dg.sh duke_msmt_to_market
```

#### Source-Free Pre-training:
```bash
cd scripts
./run_aida_sf_pretrain.sh resnet50_idm "market1501 dukemtmc msmt17"
```

#### Source-Free Adaptation:
```bash
cd scripts
./run_aida_sf_adapt.sh logs/resnet50_idm_aida_sf/pretrain_multi_source/model_best.pth.tar market1501
```

### Using Python Config Module

```python
from idm.configs import get_config, list_configs

# List available configs
print(list_configs('dg'))

# Get a config
config = get_config('market_duke_to_msmt', 'dg')
print(config)
```

### Using train_aida.py

```bash
# List all available configs
python examples/train_aida.py --list-configs

# Run a DG experiment
python examples/train_aida.py --config-type dg --config-name market_duke_to_msmt

# Run source-free pre-training
python examples/train_aida.py --config-type sf_pretrain --config-name pretrain_market_duke_msmt

# Override config values
python examples/train_aida.py --config-type dg --config-name market_duke_to_msmt \
    --override epochs=60 lr=0.0005
```

## AIDA Hyperparameters

Default values (from paper):
- `lambda_idg_0`: 1.0 (base weight for IDG loss)
- `lambda_pmr_0`: 1.0 (base weight for PMR loss)
- `lambda_sd`: 1.0 (weight for KL divergence in PMR)
- `beta_dfc`: 0.1 (scaling factor for gradient variance in DFC)
- `lambda_max`: 10.0 (maximum allowed weight value)
- `use_entropy_modulation`: True (enable entropy-based modulation)
- `entropy_weight`: 0.5 (weight for entropy modulation)

## Notes

- All AIDA configs have `USE_MS_IDG`, `USE_PMR`, and `USE_DFC` enabled by default
- Multi-source training requires datasets to be in the `data/` directory
- Source-free adaptation requires a pre-trained checkpoint from the pre-training phase
- Logs are saved to `logs/{arch}_aida_{type}/{config_name}/`

