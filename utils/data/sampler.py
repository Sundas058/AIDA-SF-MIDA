from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        # Handle both old format (_, pid, _) and new format (_, pid, _, domain_id)
        for index, sample in enumerate(data_source):
            if len(sample) >= 3:
                pid = sample[1]  # pid is always at index 1
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        # Handle both old format (_, pid, cam) and new format (_, pid, cam, domain_id)
        for index, sample in enumerate(data_source):
            if len(sample) >= 3:
                pid = sample[1]  # pid is always at index 1
                cam = sample[2]  # cam is always at index 2
            if (pid<0): continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            # Handle both old and new format
            sample = self.data_source[i]
            i_pid = sample[1]
            i_cam = sample[2]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])


        return iter(ret)


class MultiSourceSampler(Sampler):
    """
    Sampler that ensures fair sampling across multiple source domains.
    Samples identities from each domain in a round-robin fashion to ensure balanced batches.
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        
        # Group samples by domain and then by identity
        self.domain_pids = defaultdict(set)  # domain_id -> set of pids
        self.pid_domain = {}  # pid -> domain_id
        self.pid_index = defaultdict(list)  # (domain_id, pid) -> list of indices
        self.pid_cam = defaultdict(list)  # (domain_id, pid) -> list of cameras
        
        # Parse data source
        for index, sample in enumerate(data_source):
            if len(sample) >= 4:
                # New format: (fname, pid, camid, domain_id)
                pid = sample[1]
                cam = sample[2]
                domain_id = sample[3]
            else:
                # Old format: (fname, pid, camid) - default to domain 0
                pid = sample[1]
                cam = sample[2]
                domain_id = 0
            
            if pid < 0:
                continue
            
            self.domain_pids[domain_id].add(pid)
            self.pid_domain[pid] = domain_id
            self.pid_index[(domain_id, pid)].append(index)
            self.pid_cam[(domain_id, pid)].append(cam)
        
        # Get list of (domain_id, pid) pairs
        self.domain_pid_pairs = []
        for domain_id, pids in self.domain_pids.items():
            for pid in pids:
                self.domain_pid_pairs.append((domain_id, pid))
        
        self.num_samples = len(self.domain_pid_pairs) * num_instances
    
    def __len__(self):
        return self.num_samples
    
    def __iter__(self):
        # Shuffle domain-pid pairs
        indices = torch.randperm(len(self.domain_pid_pairs)).tolist()
        ret = []
        
        for idx in indices:
            domain_id, pid = self.domain_pid_pairs[idx]
            
            # Sample first instance
            available_indices = self.pid_index[(domain_id, pid)]
            i = random.choice(available_indices)
            ret.append(i)
            
            # Get camera of first instance
            sample = self.data_source[i]
            i_cam = sample[2]
            
            # Sample remaining instances from different cameras if possible
            cams = self.pid_cam[(domain_id, pid)]
            select_cams = No_index(cams, i_cam)
            
            if select_cams:
                if len(select_cams) >= self.num_instances - 1:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)
                
                for kk in cam_indexes:
                    ret.append(available_indices[kk])
            else:
                # Fall back to any available instances
                select_indices = No_index(available_indices, i)
                if not select_indices:
                    continue
                if len(select_indices) >= self.num_instances - 1:
                    ind_indexes = np.random.choice(select_indices, size=self.num_instances - 1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indices, size=self.num_instances - 1, replace=True)
                
                for kk in ind_indexes:
                    ret.append(available_indices[kk])

        return iter(ret)
