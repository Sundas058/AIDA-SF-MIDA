from __future__ import absolute_import
import os.path as osp
from collections import defaultdict
from . import create as create_dataset


class MultiSourceDataset(object):
    """
    Wrapper class that combines multiple source datasets for multi-source training.
    Each sample is extended with a domain_id field (0..K-1) to identify its source domain.
    """
    
    def __init__(self, source_names, data_dir, verbose=True, **kwargs):
        """
        Initialize multi-source dataset.
        
        Parameters
        ----------
        source_names : list of str
            List of dataset names to combine (e.g., ['market1501', 'dukemtmc', 'msmt17'])
        data_dir : str
            Root directory containing all datasets
        verbose : bool
            Whether to print dataset statistics
        **kwargs
            Additional arguments passed to individual dataset constructors
        """
        self.source_names = source_names
        self.num_sources = len(source_names)
        self.data_dir = data_dir
        
        if self.num_sources == 0:
            raise ValueError("At least one source dataset must be specified")
        
        # Load each source dataset
        self.datasets = []
        self.domain_offsets = []  # PID offset for each domain to avoid collisions
        
        for domain_id, name in enumerate(source_names):
            dataset = create_dataset(name, osp.join(data_dir, name), verbose=False, **kwargs)
            self.datasets.append(dataset)
            # Calculate PID offset: sum of all previous domains' num_train_pids
            offset = sum(d.num_train_pids for d in self.datasets[:-1])
            self.domain_offsets.append(offset)
        
        # Combine train sets with domain_id
        self.train = []
        for domain_id, dataset in enumerate(self.datasets):
            offset = self.domain_offsets[domain_id]
            for fname, pid, camid in dataset.train:
                # Shift PIDs to avoid collisions across domains
                shifted_pid = pid + offset
                # Add domain_id as 4th element: (fname, shifted_pid, camid, domain_id)
                self.train.append((fname, shifted_pid, camid, domain_id))
        
        # For query/gallery, we keep them separate per domain
        # The first dataset's query/gallery is used as default (for backward compatibility)
        self.query = self.datasets[0].query
        self.gallery = self.datasets[0].gallery
        
        # Calculate combined statistics
        self.num_train_pids = sum(d.num_train_pids for d in self.datasets)
        self.num_train_imgs = len(self.train)
        self.num_train_cams = max(d.num_train_cams for d in self.datasets)
        
        # Use first dataset's query/gallery stats (for backward compatibility)
        self.num_query_pids = self.datasets[0].num_query_pids
        self.num_query_imgs = len(self.query)
        self.num_query_cams = self.datasets[0].num_query_cams
        self.num_gallery_pids = self.datasets[0].num_gallery_pids
        self.num_gallery_imgs = len(self.gallery)
        self.num_gallery_cams = self.datasets[0].num_gallery_cams
        
        # Store images_dir from first dataset (for backward compatibility)
        self.images_dir = self.datasets[0].images_dir
        
        if verbose:
            print("=> MultiSourceDataset loaded:")
            print("  Sources: {}".format(source_names))
            print("  Total train identities: {:5d}".format(self.num_train_pids))
            print("  Total train images: {:8d}".format(self.num_train_imgs))
            print("  Domain breakdown:")
            for domain_id, (name, dataset) in enumerate(zip(source_names, self.datasets)):
                domain_samples = sum(1 for _, _, _, d_id in self.train if d_id == domain_id)
                print("    Domain {} ({}): {:5d} ids, {:8d} images".format(
                    domain_id, name, dataset.num_train_pids, domain_samples))
    
    def get_domain_samples(self, domain_id):
        """Get all samples from a specific domain."""
        return [(fname, pid, camid) for fname, pid, camid, d_id in self.train if d_id == domain_id]
    
    def get_domain_info(self, domain_id):
        """Get dataset info for a specific domain."""
        return self.datasets[domain_id]

