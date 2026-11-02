"""Market1501 dataset adapter."""
from pathlib import Path
from .reid import load_dataset

class Market1501:
    def __init__(self, root=None, verbose=True, **kwargs):
        root = Path(root or "data/market1501").resolve()
        dataset = load_dataset("market1501", root.parent,
                               splits=kwargs.get("splits", ("train", "query", "gallery")),
                               unlabeled_train=kwargs.get("unlabeled_train", False))
        self.__dict__.update(dataset.__dict__)
        for split in ("train", "query", "gallery"):
            rows = getattr(dataset, split)
            setattr(self, f"num_{split}_pids", len({row[1] for row in rows}))
            setattr(self, f"num_{split}_imgs", len(rows))
            setattr(self, f"num_{split}_cams", len({row[2] for row in rows}))

