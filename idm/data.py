"""Identity-balanced batches with all source domains represented."""
import random
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms as T


class ImageRows(Dataset):
    def __init__(self, rows, training=False, height=256, width=128, erase=0.5):
        self.rows = rows
        ops = [T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC)]
        if training:
            ops += [T.RandomHorizontalFlip()]
        ops += [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        if training and erase:
            ops += [T.RandomErasing(p=erase, value=0)]
        self.transform = T.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, pid, camera, *domain = self.rows[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path, pid, camera, domain[0] if domain else 0, index


class DomainIdentityBatchSampler(Sampler):
    def __init__(self, rows, batch_size=64, instances=4, steps=400, seed=1):
        if batch_size % instances or batch_size < instances * 2 or instances < 2:
            raise ValueError("Batch size must be divisible by instances and contain at least two identities.")
        self.groups = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(rows):
            domain = row[3] if len(row) > 3 else 0
            self.groups[domain][row[1]].append(index)
        self.domains = sorted(self.groups)
        self.identities = batch_size // instances
        if self.identities < len(self.domains):
            raise ValueError("Batch must contain at least one identity per domain.")
        if sum(len(group) for group in self.groups.values()) < 2:
            raise ValueError("Training requires at least two identity clusters.")
        self.instances, self.steps, self.seed, self.epoch = instances, steps, seed, 0

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for step in range(self.steps):
            batch = []
            selected = defaultdict(set)
            for slot in range(self.identities):
                domain = self.domains[(slot + step) % len(self.domains)]
                available = [pid for pid in self.groups[domain] if pid not in selected[domain]]
                pid = rng.choice(available or list(self.groups[domain]))
                selected[domain].add(pid)
                indices = self.groups[domain][pid]
                batch += rng.sample(indices, self.instances) if len(indices) >= self.instances else rng.choices(indices, k=self.instances)
            yield batch


def image_loader(rows, config, training=False, seed=1):
    dataset = ImageRows(rows, training, config["height"], config["width"], config["random_erasing"])
    if training:
        sampler = DomainIdentityBatchSampler(rows, config["batch_size"], config["num_instances"], config["iters"], seed)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=config["workers"], pin_memory=torch.cuda.is_available())
    return DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["workers"], shuffle=False)


def merge_sources(datasets):
    rows, offset = [], 0
    for domain, dataset in enumerate(datasets):
        for path, pid, camera in dataset.train:
            rows.append((path, pid + offset, camera, domain))
        offset += dataset.num_train_pids
    return rows, offset

