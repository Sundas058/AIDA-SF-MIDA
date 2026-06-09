from __future__ import print_function, absolute_import
import os
import os.path as osp
import glob
import re

# 🔹 Absolute imports from IDM repo
from idm.utils.data import BaseImageDataset


class DukeMTMC(BaseImageDataset):
    """
    DukeMTMC-reID Dataset
    Reference:
    1. Ristani et al. ECCVW 2016.
    2. Zheng et al. ICCV 2017.
    URL: https://github.com/layumi/DukeMTMC-reID_evaluation

    Dataset statistics:
    # identities: 1404 (train + query)
    # images: 16522 (train) + 2228 (query) + 17661 (gallery)
    # cameras: 8
    """

    # Path to your dataset
    DATASET_ROOT = "/home/tq_sundas/IDM-main/idm/datasets"
    DATASET_NAME = "DukeMTMC-reID"

    def __init__(self, verbose=True, **kwargs):
        super(DukeMTMC, self).__init__()

        root = kwargs.get('root', osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))))
        self.dataset_dir = osp.join(root, 'DukeMTMC-reID')
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        # Check that directories exist
        self._check_before_run()

        # Process dataset directories
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> DukeMTMC-reID loaded from: {}".format(self.dataset_dir))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all required directories exist"""
        required_dirs = [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir]
        for d in required_dirs:
            if not osp.exists(d):
                raise RuntimeError("'{}' is not available".format(d))

    def _process_dir(self, dir_path, relabel=False):
        """Process a directory and return list of (img_path, pid, camid)"""
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(osp.basename(img_path)).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(osp.basename(img_path)).groups())
            assert 1 <= camid <= 8
            camid -= 1  # zero-based indexing
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
