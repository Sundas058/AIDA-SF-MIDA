"""Deterministic dataset readers with explicit train/query/gallery splits."""
import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReIDDataset:
    train: list
    query: list
    gallery: list
    images_dir: str = ""

    def __getattr__(self, name):
        for split in ("train", "query", "gallery"):
            rows = object.__getattribute__(self, split)
            if name == f"num_{split}_imgs":
                return len(rows)
            if name == f"num_{split}_pids":
                return len({r[1] for r in rows})
            if name == f"num_{split}_cams":
                return len({r[2] for r in rows})
        raise AttributeError(name)

    @property
    def num_train_pids(self):
        return len({r[1] for r in self.train})

    @property
    def num_train_cams(self):
        return len({r[2] for r in self.train})


def _relabel(rows):
    mapping = {pid: i for i, pid in enumerate(sorted({r[1] for r in rows}))}
    return [(path, mapping[pid], cam) for path, pid, cam in rows]


def _read_directory(directory, relabel=False, unlabeled=False):
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {directory}")
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        match = re.search(r"(-?\d+)_c(\d+)", path.name)
        if match:
            pid, camera = map(int, match.groups())
            if pid == -1:
                continue
            rows.append((str(path.resolve()), -1 if unlabeled else pid, camera - 1))
        elif unlabeled:
            rows.append((str(path.resolve()), -1, 0))
        else:
            raise ValueError(f"Cannot read identity/camera from {path.name}; use a split CSV.")
    if not rows:
        raise ValueError(f"No images found in {directory}")
    return _relabel(rows) if relabel and not unlabeled else rows


def _read_manifest(path, relabel=False, unlabeled=False):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "pid", "camid"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: columns must include path,pid,camid")
        for row in reader:
            image = (path.parent / row["path"]).resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            rows.append((str(image), -1 if unlabeled else int(row["pid"]), int(row["camid"])))
    if not rows:
        raise ValueError(f"Empty split: {path}")
    return _relabel(rows) if relabel and not unlabeled else rows


def _msmt_split(base, split, unlabeled=False):
    rows = []
    files = ["list_train.txt", "list_val.txt"] if split == "train" else [f"list_{split}.txt"]
    for filename in files:
        listing = base / filename
        if filename == "list_val.txt" and not listing.exists():
            continue
        with listing.open(encoding="utf-8") as stream:
            for line in stream:
                relative, label = line.split()
                name = Path(relative).name
                camera = int(name.split("_")[2]) - 1
                part = ("mask_train_v2" if split == "train" else "mask_test_v2") if base.name == "MSMT17_V2" else ("train" if split == "train" else "test")
                image = base / relative
                if not image.is_file():
                    image = base / part / relative
                if not image.is_file():
                    raise FileNotFoundError(image)
                rows.append((str(image.resolve()), -1 if unlabeled else int(label), camera))
    return _relabel(rows) if split == "train" and not unlabeled else rows


def load_dataset(name, data_dir, splits=("train", "query", "gallery"), unlabeled_train=False):
    root = Path(data_dir).resolve() / name
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    candidates = [root] + sorted(p for p in root.iterdir() if p.is_dir())
    if name == "msmt17":
        base = next((p for p in candidates if (p / "list_train.txt").is_file()), root)
    else:
        base = next((p for p in candidates if (p / "bounding_box_train").is_dir() or (p / "train.csv").is_file()), root)
    result = {"train": [], "query": [], "gallery": []}
    for split in splits:
        manifest = base / f"{split}.csv"
        unlabeled = split == "train" and unlabeled_train
        if manifest.is_file():
            result[split] = _read_manifest(manifest, split == "train", unlabeled)
        elif name == "msmt17":
            result[split] = _msmt_split(base, split, unlabeled)
        else:
            folder = {"train": "bounding_box_train", "query": "query", "gallery": "bounding_box_test"}[split]
            result[split] = _read_directory(base / folder, split == "train", unlabeled)
    return ReIDDataset(**result)

