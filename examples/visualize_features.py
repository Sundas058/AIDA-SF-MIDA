"""Extract checkpoint embeddings and draw a seeded t-SNE feature map."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from idm.engine import load_checkpoint, make_model, resolve_device, seed_everything
from idm.data import image_loader
from idm.datasets.reid import load_dataset
from idm.evaluators import extract_features
from idm.models.domain_bn import set_domain

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="runs/features.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    if args.limit < 3:
        parser.error("--limit must be at least 3")
    checkpoint = load_checkpoint(args.checkpoint)
    config = {**checkpoint["config"], "workers": args.workers}
    seed_everything(config["seed"])
    model = make_model(config, checkpoint["num_classes"], resolve_device(args.device),
                       pretrained=False, uda=checkpoint["mode"] == "uda")
    model.load_state_dict(checkpoint["state_dict"])
    set_domain(model, "target")
    dataset = load_dataset(args.target, args.data_dir, splits=("query", "gallery"))
    rows = list(dict.fromkeys(dataset.query + dataset.gallery))
    if len(rows) > args.limit:
        chosen = np.random.default_rng(config["seed"]).choice(len(rows), args.limit, replace=False)
        rows = [rows[i] for i in sorted(chosen)]
    if len(rows) < 3:
        parser.error("At least three evaluation images are required for t-SNE.")
    features, _ = extract_features(model, image_loader(rows, config))
    matrix = torch.stack([features[row[0]] for row in rows]).numpy()
    labels = np.asarray([row[1] for row in rows])
    embedding = TSNE(n_components=2, perplexity=min(30, (len(rows) - 1) / 3),
                     init="pca", learning_rate="auto", random_state=config["seed"]).fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab20", s=14, alpha=0.8, linewidths=0)
    ax.set_title(f"AIDA-ReID · {args.target} · t-SNE", loc="left", weight="bold", pad=15)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.spines[["top", "right"]].set_visible(False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    np.savez_compressed(output.with_suffix(".npz"), features=matrix, labels=labels,
                        paths=np.asarray([row[0] for row in rows]), tsne=embedding)
    print(f"Saved {output} and {output.with_suffix('.npz')}")

if __name__ == "__main__":
    main()

