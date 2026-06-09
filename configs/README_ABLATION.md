# AIDA Ablation Study Configurations

This document describes how to run ablation studies with different combinations of AIDA modules.

## Available Ablation Configurations

1. **`idmpp_baseline`**: IDM++ baseline (all AIDA flags False)
   - No MS-IDG, no PMR, no DFC
   - Equivalent to original IDM++

2. **`idmpp_ms_idg`**: IDM++ + MS-IDG
   - Multi-Source Intermediate Domain Generator enabled
   - PMR and DFC disabled

3. **`idmpp_ms_idg_pmr`**: IDM++ + MS-IDG + PMR
   - MS-IDG and Pseudo-Mirror Regularization enabled
   - DFC disabled

4. **`aida_full`**: Full AIDA/SF-MIDA
   - All modules enabled: MS-IDG + PMR + DFC
   - Complete AIDA implementation

## Usage

### Using Shell Script

```bash
# Run ablation study
./scripts/run_ablation.sh <source> <target> <ablation> [arch]

# Examples:
./scripts/run_ablation.sh market1501 dukemtmc idmpp_baseline
./scripts/run_ablation.sh market1501 dukemtmc idmpp_ms_idg
./scripts/run_ablation.sh market1501 dukemtmc idmpp_ms_idg_pmr
./scripts/run_ablation.sh market1501 dukemtmc aida_full
```

### Using Python Config Module

```python
from idm.configs import apply_ablation_to_args, get_ablation_description

# Get description
print(get_ablation_description('aida_full'))

# Apply to args (in your training script)
args, description = apply_ablation_to_args(args, 'aida_full')
print(f"Running: {description}")
```

### Using in Training Script

You can also use ablation configs directly in your training script:

```python
from idm.configs import get_ablation_config

# Get ablation config
ablation = get_ablation_config('idmpp_ms_idg_pmr')

# Apply to your args
for key, value in ablation.items():
    if key != 'description':
        setattr(args, key, value)
```

## Extended Evaluation Metrics

All ablation experiments automatically compute extended metrics when `--compute-extended-metrics` is enabled:

- **CMC**: Cumulative Matching Characteristics (Rank-1, Rank-5, Rank-10)
- **mAP**: Mean Average Precision
- **NMI**: Normalized Mutual Information (clustering quality)
- **Silhouette**: Silhouette Score (feature clustering quality)
- **FID**: Fréchet Inception Distance (optional, requires `--compute-fid`)

### Enabling Extended Metrics

```bash
# In training script (already enabled in run_ablation.sh)
--compute-extended-metrics

# Optional: also compute FID (slower)
--compute-extended-metrics --compute-fid
```

## Output

Results are saved in:
```
logs/{arch}_ablation/{source}-TO-{target}/{ablation_name}/
```

Each experiment logs:
- Standard metrics: Rank-1, Rank-5, Rank-10, mAP
- Extended metrics: NMI, Silhouette, (optionally FID)
- Training logs with AIDA loss components

## Example Ablation Study Workflow

```bash
# 1. Baseline
./scripts/run_ablation.sh market1501 dukemtmc idmpp_baseline

# 2. +MS-IDG
./scripts/run_ablation.sh market1501 dukemtmc idmpp_ms_idg

# 3. +MS-IDG + PMR
./scripts/run_ablation.sh market1501 dukemtmc idmpp_ms_idg_pmr

# 4. Full AIDA
./scripts/run_ablation.sh market1501 dukemtmc aida_full

# Compare results
# All logs are in logs/resnet50_idm_ablation/market1501-TO-dukemtmc/
```

## Notes

- All ablation configs use the same base hyperparameters (mu1, mu2, mu3, etc.)
- Only AIDA flags differ between ablations
- Extended metrics are computed automatically to enable comprehensive comparison
- Backward compatibility: if extended metrics are not requested, evaluation behaves as before

