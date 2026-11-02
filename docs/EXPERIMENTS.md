# 🔬 Experiment guide

Run commands from the repository root. All configuration values are saved in each run directory.

## Domain generalization

| Preset | Labeled sources | Unseen target |
|:--|:--|:--|
| `market_duke_to_msmt` | Market + Duke | MSMT17 |
| `market_msmt_to_duke` | Market + MSMT17 | Duke |
| `duke_msmt_to_market` | Duke + MSMT17 | Market |
| `market_duke_msmt_to_cuhk03` | Market + Duke + MSMT17 | CUHK03 |
| `market_duke_msmt_to_personx` | Market + Duke + MSMT17 | PersonX |
| `personx_to_market` | PersonX | Market |
| `personx_to_duke` | PersonX | Duke |
| `personx_to_msmt` | PersonX | MSMT17 |

```bash
python examples/train_aida.py --config-type dg --config-name market_msmt_to_duke --data-dir data --logs-dir runs/dg-duke
python examples/test.py --checkpoint runs/dg-duke/checkpoint.pth --target dukemtmc --data-dir data --output runs/dg-duke/metrics.json
```

Training never opens the configured DG target. Evaluate the final checkpoint separately; target validation scores are not used to choose a checkpoint.

## Source-free pretraining pairs

| Target | Pretraining preset | Adaptation preset |
|:--|:--|:--|
| Market | `pretrain_for_market` | `adapt_to_market` |
| Duke | `pretrain_for_duke` | `adapt_to_duke` |
| MSMT17 | `pretrain_for_msmt` | `adapt_to_msmt` |

```bash
python examples/train_aida.py --config-type sf_pretrain --config-name pretrain_for_msmt --logs-dir runs/pretrain-msmt
python examples/train_aida.py --config-type sf_adapt --config-name adapt_to_msmt --pretrained-checkpoint runs/pretrain-msmt/checkpoint.pth --logs-dir runs/sf-msmt
```

The target-only `--data-dir` may contain just the target dataset. The checkpoint supplies the embedding architecture and source-domain provenance.

## UDA

The `uda` group contains every directed pair among `market1501`, `dukemtmc`, and `msmt17`.

```bash
python examples/train_idm.py --config-name dukemtmc_to_market1501 --data-dir data --logs-dir runs/uda-duke-market
```

For UDA, batch size is the total source-plus-target count. The default 64 uses 32 source and 32 pseudo-labeled target images. Each half must be divisible by the configured instances per identity.

## Component switches

| Ablation | MS-IDG | PMR identity | PMR relational | DFC |
|:--|:--:|:--:|:--:|:--:|
| `source_baseline` | — | — | — | — |
| `ms_idg` | ✓ | — | — | — |
| `ms_idg_identity` | ✓ | ✓ | — | — |
| `ms_idg_relational` | ✓ | — | ✓ | — |
| `ms_idg_pmr` | ✓ | ✓ | ✓ | — |
| `aida_full` | ✓ | ✓ | ✓ | ✓ |

`source_baseline` is the supervised source control. It is distinct from the pairwise IDM/IDM++ UDA path and should retain that name in result tables.

```bash
python examples/train_aida.py --config-name market_duke_to_msmt --ablation ms_idg_pmr --override seed=1 --logs-dir runs/msidg-pmr-seed1
python examples/train_baseline.py --config-name market_duke_to_msmt --override seed=1 --logs-dir runs/source-control-seed1
```

Hold initialization, source splits, seed, augmentations, schedule, and evaluation protocol constant for a controlled comparison. The manuscript's fixed-run ablation measurements and its multi-run main results use separate experimental configurations; keep each experiment's resolved configuration with its metrics.

## Multiple seeds

PowerShell:

```powershell
foreach ($seed in 1, 2, 3) {
    python examples/train_aida.py --config-name market_duke_to_msmt --override "seed=$seed" --logs-dir "runs/msmt-seed-$seed"
    python examples/test.py --checkpoint "runs/msmt-seed-$seed/checkpoint.pth" --target msmt17 --output "runs/msmt-seed-$seed/metrics.json"
}
python examples/analyze_results.py --inputs runs/msmt-seed-1/metrics.json runs/msmt-seed-2/metrics.json runs/msmt-seed-3/metrics.json --output runs/msmt-summary.json
```

The analysis command reports the mean and sample standard deviation for Rank-1/5/10 and mAP. With `--baseline`, it pairs run files in the supplied order and reports a paired t-test.

## Continue a run

Use the same experiment settings, output directory, and epoch budget:

```bash
python examples/train_aida.py --config-name market_duke_to_msmt --logs-dir runs/dg-msmt --resume runs/dg-msmt/checkpoint.pth
```

The checkpoint contains model parameters, classifier size, optimizer/scheduler state, controller history, source-domain names, and random-number states. Source-free runs retain only target-side optimizer state and pretrained model parameters.

## Feature visualization

```bash
python examples/visualize_features.py --checkpoint runs/dg-msmt/checkpoint.pth --target msmt17 --data-dir data --output runs/dg-msmt/features.png --limit 2000
```

This command writes both a t-SNE figure and the extracted features in a companion `.npz` file. It uses evaluation images and the checkpoint's embedding architecture.

## Diagnostics

- Use `--dry-run` to inspect settings without opening datasets or downloading weights.
- On Windows, `--override workers=0` makes data-loading errors easier to inspect.
- DBSCAN must produce at least two usable identity clusters. An explicit error stops training if the current embeddings do not satisfy this condition.
- Dense k-reciprocal/Jaccard clustering has quadratic memory cost in target image count. Keep the original training split and provision sufficient host memory.
- Initial source training uses torchvision's ImageNet weights; put those weights in the torchvision cache for offline runs.

