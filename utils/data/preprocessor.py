from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
import torch
import pdb


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        # Handle both old format (fname, pid, camid) and new format (fname, pid, camid, domain_id)
        sample = self.dataset[index]
        if len(sample) == 4:
            fname, pid, camid, domain_id = sample
        else:
            fname, pid, camid = sample
            domain_id = 0  # Default domain_id for backward compatibility
        
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # Return domain_id as 5th element: (img, fname, pid, camid, domain_id, index)
        return img, fname, pid, camid, domain_id, index


