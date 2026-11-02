"""Dataset factory. Each root is the directory for the named dataset."""
from pathlib import Path
from .reid import load_dataset
def names():
    return ["cuhk03", "dukemtmc", "market1501", "msmt17", "personx", "unreal"]
def create(name, root, *args, **kwargs):
    if name not in names():
        raise KeyError(f"Unknown dataset: {name}")
    return load_dataset(name, Path(root).resolve().parent,
                        splits=kwargs.get("splits", ("train", "query", "gallery")),
                        unlabeled_train=kwargs.get("unlabeled_train", False))
get_dataset = create

