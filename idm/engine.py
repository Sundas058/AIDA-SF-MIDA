"""CLI orchestration for Algorithm 1, checkpointing, and retrieval evaluation."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from .configs.aida_configs import GROUPS, get_config
from .configs.ablation_configs import ABLATION_CONFIGS, get_ablation_config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def validate_config(config, mode, checkpoint=None):
    if config["target"] in config["sources"]:
        raise ValueError("Target must be excluded from labeled source training.")
    if mode == "sf_adapt":
        if config["sources"]:
            raise ValueError("Source-free refinement must have an empty source list.")
        if not checkpoint:
            raise ValueError("Source-free refinement requires --pretrained-checkpoint or --resume.")
    elif not config["sources"]:
        raise ValueError("Source training requires at least one source.")
    if mode == "uda" and len(config["sources"]) != 1:
        raise ValueError("Single-source UDA requires exactly one labeled source.")
    if mode == "uda" and (config["num_instances"] < 2 or config["batch_size"] % (2 * config["num_instances"])):
        raise ValueError("UDA batch size must be divisible by twice num_instances.")
    if config["epochs"] < 1 or not 0 <= config["supervised_epochs"] <= config["epochs"]:
        raise ValueError("Invalid supervised/total epoch counts.")
    if config["iters"] < 1 or config["workers"] < 0 or config["lr"] <= 0:
        raise ValueError("Iterations and learning rate must be positive; workers must be nonnegative.")
    if not 0 <= config["random_erasing"] <= 1 or config["height"] < 32 or config["width"] < 16:
        raise ValueError("Invalid augmentation probability or input dimensions.")
    if config["num_instances"] < 2 or config["batch_size"] % config["num_instances"]:
        raise ValueError("Batch size must be divisible by num_instances (at least 2).")
    if config["batch_size"] // config["num_instances"] < max(2, len(config["sources"])):
        raise ValueError("Batch has too few identity groups for the source domains.")
    if mode != "sf_adapt" and config["use_pmr"] and not config["use_ms_idg"]:
        raise ValueError("Source PMR requires MS-IDG.")
    if len(set(config["sources"])) != len(config["sources"]):
        raise ValueError("Source domains must be unique.")
    if config["num_parts"] < 0 or config["memory_slots"] < 0 or config["sam_rho"] < 0:
        raise ValueError("Parts, memory slots, and SAM radius must be nonnegative.")
    if config["min_samples"] < 2 or config["eps"] <= 0 or config["k1"] < 1 or config["k2"] < 1:
        raise ValueError("Invalid DBSCAN/Jaccard parameters.")
    return config


def make_model(config, num_classes, device, pretrained=None, uda=False):
    from .models.aida import AIDAEmbedding
    from .models.domain_bn import convert_domain_bn
    model = AIDAEmbedding(num_classes, config["pretrained"] if pretrained is None else pretrained,
                          config["use_gem"], config["num_parts"], config["memory_slots"], pairwise_idm=uda)
    if uda:
        convert_domain_bn(model)
    return model.to(device)


def make_optimizer(model, config):
    from .utils.optim_sam import SAM
    base = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    optimizer = SAM(model.parameters(), base, rho=config["sam_rho"]) if config["use_sam"] else base
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base, T_max=config["epochs"])
    return optimizer, base, scheduler


def replace_classifier(model, centers, optimizer=None):
    old = set(model.classifier.parameters())
    model.reset_classifier(centers)
    if optimizer is not None:
        base = getattr(optimizer, "base_optimizer", optimizer)
        for parameter in old:
            base.state.pop(parameter, None)
            optimizer.state.pop(parameter, None)
        for group in base.param_groups:
            group["params"] = [p for p in group["params"] if p not in old]
        base.param_groups[0]["params"] += list(model.classifier.parameters())
        # SAM wraps the same base parameters but owns a separate parameter-group list.
        if base is not optimizer:
            for group in optimizer.param_groups:
                group["params"] = [p for p in group["params"] if p not in old]
            optimizer.param_groups[0]["params"] += list(model.classifier.parameters())


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def save_checkpoint(path, model, config, mode, epoch, controller, base, scheduler, provenance):
    state = {"state_dict": model.state_dict(), "config": config, "mode": mode, "epoch": epoch,
             "num_classes": model.classifier.out_features, "controller": controller.state_dict(),
             "optimizer": base.state_dict(), "scheduler": scheduler.state_dict(),
             "source_domains": provenance, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "python_rng": random.getstate()}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run_training(config, mode, checkpoint_path=None, resume=None):
    from .datasets.reid import load_dataset
    from .data import image_loader, merge_sources
    from .evaluators import extract_features
    from .clustering import cluster_target
    from .trainers import AIDATrainer
    from .utils.dfc import DynamicFeedbackController
    from .models.domain_bn import set_domain

    validate_config(config, mode, resume or checkpoint_path)
    seed_everything(config["seed"])
    device = resolve_device(config["device"])
    logs = Path(config["logs_dir"])
    if (logs / "checkpoint.pth").exists() and not resume:
        raise ValueError("Output directory already contains a checkpoint; use --resume or select a fresh --logs-dir.")
    logs.mkdir(parents=True, exist_ok=True)
    initial = load_checkpoint(resume or checkpoint_path) if (resume or checkpoint_path) else None
    if resume and (initial["mode"] != mode or initial["config"]["sources"] != config["sources"] or
                   initial["config"]["target"] != config["target"]):
        raise ValueError("Resume checkpoint protocol, sources, and target must match the selected experiment.")
    if resume:
        runtime_keys = {"data_dir", "logs_dir", "device", "workers"}
        if mode == "sf_adapt":
            runtime_keys.update({"use_gem", "num_parts", "memory_slots"})
        changed = [key for key in config if key not in runtime_keys and
                   config[key] != initial["config"].get(key)]
        if changed:
            raise ValueError(f"Resume configuration differs from checkpoint: {', '.join(changed)}")
    if mode == "sf_adapt":
        provenance = initial.get("source_domains")
        if not provenance:
            raise ValueError("Pretraining checkpoint must record its labeled source domains.")
        if config["target"] in provenance:
            raise ValueError("Target leakage: this checkpoint was pretrained on the selected target.")
        if initial["mode"] == "uda":
            raise ValueError("SF-MIDA requires a DG/source-pretraining checkpoint.")
        for key in ("use_gem", "num_parts", "memory_slots"):
            config[key] = initial["config"][key]
        source_rows, source_classes = [], 0
    else:
        provenance = list(config["sources"])
        # DG and SF pretraining deliberately load no target split here.
        datasets = [load_dataset(name, config["data_dir"], splits=("train",)) for name in config["sources"]]
        source_rows, source_classes = merge_sources(datasets)
    target_data = None
    if mode in ("sf_adapt", "uda"):
        target_data = load_dataset(config["target"], config["data_dir"], splits=("train",), unlabeled_train=True)
    nclasses = initial["num_classes"] if initial else source_classes
    model = make_model(config, nclasses, device, pretrained=False if initial else None, uda=mode == "uda")
    if initial:
        model.load_state_dict(initial["state_dict"], strict=True)
    controller = DynamicFeedbackController(
        num_sources=2 if mode == "uda" else max(1, len(config["sources"])),
        lambda_idg_0=config["lambda_idg_0"], lambda_pmr_0=config["lambda_pmr_0"],
        beta=config["beta_dfc"], gamma=config["gamma_dfc"], lambda_max=config["lambda_max"],
        entropy_max=config["entropy_max"], window_size=config["grad_window"], scale_max=config["scale_max"])
    optimizer, base, scheduler = make_optimizer(model, config)
    start_epoch = 0
    if resume:
        base.load_state_dict(initial["optimizer"])
        scheduler.load_state_dict(initial["scheduler"])
        controller.load_state_dict(initial["controller"])
        start_epoch = initial["epoch"]
        torch.set_rng_state(initial["torch_rng"])
        random.setstate(initial["python_rng"])
        if torch.cuda.is_available() and initial["cuda_rng"]:
            torch.cuda.set_rng_state_all(initial["cuda_rng"])
    (logs / "config.json").write_text(json.dumps({**config, "mode": mode}, indent=2), encoding="utf-8")
    trainer = AIDATrainer(model, config, controller)
    batch_config = {**config, "batch_size": config["batch_size"] // 2} if mode == "uda" else config
    source_loader = image_loader(source_rows, batch_config, True, config["seed"]) if source_rows else None
    for epoch in range(start_epoch, config["epochs"]):
        stage = "sf" if mode == "sf_adapt" else ("supervised" if epoch < config["supervised_epochs"] else "aida")
        cluster_info = {}
        target_loader = None
        if target_data is not None and (mode == "sf_adapt" or stage != "supervised"):
            set_domain(model, "target")
            features, _ = extract_features(model, image_loader(target_data.train, config))
            ordered = torch.stack([features[row[0]] for row in target_data.train])
            labels, centers = cluster_target(ordered, config["eps"], config["min_samples"], config["k1"], config["k2"])
            if mode == "uda":
                # Source weights remain fixed in their class slots; target slots are epoch-specific.
                centers = torch.cat([model.classifier.weight[:source_classes].detach().cpu(), centers])
            replace_classifier(model, centers.to(device), optimizer)
            pseudo_rows = [(path, int(label) + source_classes, camera, 0) for
                           (path, _, camera), label in zip(target_data.train, labels) if label >= 0]
            target_loader = image_loader(pseudo_rows, batch_config, True, config["seed"] + epoch)
            cluster_info = {"clusters": len(set(labels) - {-1}), "noise": int((labels < 0).sum())}
        loader = target_loader if mode == "sf_adapt" else source_loader
        loader.batch_sampler.epoch = epoch
        target_iterator = iter(target_loader) if mode == "uda" and target_loader else None
        for step, batch in enumerate(loader):
            set_domain(model, "source" if mode == "uda" else "target")
            values = trainer.step(batch, optimizer, stage,
                target_batch=next(target_iterator) if target_iterator is not None else None)
            record = {"epoch": epoch + 1, "step": step + 1, "stage": stage, **values, **cluster_info}
            with (logs / "train.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            if step == 0 or (step + 1) % 50 == 0:
                print(f"epoch={epoch+1}/{config['epochs']} step={step+1} stage={stage} loss={values['loss']:.4f}", flush=True)
        scheduler.step()
        save_checkpoint(logs / "checkpoint.pth", model, config, mode, epoch + 1, controller, base, scheduler, provenance)
    print(f"Checkpoint: {logs / 'checkpoint.pth'}")
    return logs / "checkpoint.pth"


def run_evaluation(checkpoint_path, target, data_dir, output, device="auto", workers=4, batch_size=64, extended=False):
    from .datasets.reid import load_dataset
    from .data import image_loader
    from .evaluators import extract_features, retrieval_metrics
    from .models.domain_bn import set_domain
    checkpoint = load_checkpoint(checkpoint_path)
    config = {**checkpoint["config"], "workers": workers, "batch_size": batch_size}
    seed_everything(config["seed"])
    model = make_model(config, checkpoint["num_classes"], resolve_device(device), pretrained=False, uda=checkpoint["mode"] == "uda")
    model.load_state_dict(checkpoint["state_dict"])
    set_domain(model, "target")
    dataset = load_dataset(target, data_dir, splits=("query", "gallery"))
    rows = list(dict.fromkeys(dataset.query + dataset.gallery))
    features, _ = extract_features(model, image_loader(rows, config))
    q = torch.stack([features[row[0]] for row in dataset.query])
    g = torch.stack([features[row[0]] for row in dataset.gallery])
    metrics = retrieval_metrics(q, g, dataset.query, dataset.gallery)
    if extended:
        from .evaluation_metrics.extended_metrics import compute_nmi, compute_silhouette
        array = torch.stack([features[row[0]] for row in rows]).numpy()
        identities = np.asarray([row[1] for row in rows])
        metrics["NMI"] = float(compute_nmi(array, identities))
        metrics["silhouette"] = float(compute_silhouette(array, identities, sample_size=min(5000, len(array))))
    metrics.update(target=target, checkpoint=str(Path(checkpoint_path).resolve()), seed=config["seed"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def train_main(argv=None, default_mode="dg", baseline=False):
    parser = argparse.ArgumentParser(description="AIDA-ReID / SF-MIDA training")
    parser.add_argument("--config-type", choices=sorted(GROUPS), default=default_mode)
    parser.add_argument("--config-name")
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--data-dir")
    parser.add_argument("--logs-dir")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--ablation", choices=sorted(ABLATION_CONFIGS))
    parser.add_argument("--override", nargs="+", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the resolved configuration.")
    args = parser.parse_args(argv)
    if args.list_configs:
        print(json.dumps({group: sorted(values) for group, values in GROUPS.items()}, indent=2))
        return
    if not args.config_name:
        parser.error("--config-name is required unless --list-configs is used")
    try:
        config = get_config(args.config_name, args.config_type)
        if args.ablation:
            config.update(get_ablation_config(args.ablation))
        if baseline:
            config.update(use_ms_idg=False, use_pmr=False, use_dfc=False)
        for override in args.override:
            if "=" not in override:
                raise ValueError(f"Expected KEY=VALUE: {override}")
            key, raw = override.split("=", 1)
            if key not in config:
                raise ValueError(f"Unknown configuration key: {key}")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            original = config[key]
            if isinstance(original, bool) and not isinstance(value, bool):
                raise ValueError(f"{key} requires true or false")
            if isinstance(original, int) and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"{key} requires an integer")
            if isinstance(original, float) and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                raise ValueError(f"{key} requires a number")
            if isinstance(original, list) and not isinstance(value, list):
                raise ValueError(f"{key} requires a JSON array")
            config[key] = value
        if args.data_dir:
            config["data_dir"] = args.data_dir
        config["logs_dir"] = args.logs_dir or config["logs_dir"] or str(Path("runs") / args.config_type / args.config_name)
        validate_config(config, args.config_type, args.resume or args.pretrained_checkpoint)
        if args.pretrained_checkpoint and args.config_type != "sf_adapt":
            raise ValueError("--pretrained-checkpoint is reserved for source-free refinement; use --resume to continue a run.")
        if args.resume and args.pretrained_checkpoint:
            raise ValueError("Choose either --resume or --pretrained-checkpoint.")
        if args.dry_run:
            print(json.dumps({**config, "mode": args.config_type}, indent=2))
            return
        run_training(config, args.config_type, args.pretrained_checkpoint, args.resume)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate cross-camera CMC Rank-1/5/10 and mAP.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="runs/evaluation.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args(argv)
    run_evaluation(args.checkpoint, args.target, args.data_dir, args.output, args.device, args.workers, args.batch_size, args.extended)

