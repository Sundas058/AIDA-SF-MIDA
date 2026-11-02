# ⚙️ Configuration

Main presets are in [aida_configs.py](aida_configs.py), with full commands in [the experiment guide](../../docs/EXPERIMENTS.md).

```bash
python examples/train_aida.py --list-configs
python examples/train_aida.py --config-name market_duke_to_msmt --dry-run
```

Groups: `dg`, `sf_pretrain`, `sf_adapt`, and `uda`. Configurations are copied before overrides, so one run does not mutate another preset.

Override typed values with `--override key=value`. Unknown keys and incompatible basic types are rejected. Use JSON booleans: `true` / `false`.

Source-free adaptation requires `--pretrained-checkpoint`; continuing an existing run uses `--resume`. Those arguments are mutually exclusive.

