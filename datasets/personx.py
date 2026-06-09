from __future__ import print_function, absolute_import
import os
import os.path as osp
import glob
import re

# 🔹 Fix for 'idm' module not found
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from idm.utils.data import BaseImageDataset
from idm.utils.osutils import mkdir_if_missing
from idm.utils.serialization import write_json

class PersonX(BaseImageDataset):
    """
    PersonX Dataset
    Reference:
    Sun et al. Dissecting Person Re-identification from the Viewpoint of Viewpoint. CVPR 2019.

    Dataset statistics:
    # identities: 1266
    # images: 9840 (train) + 5136 (query) + 30816 (gallery)
    """

    # 🔹 Set your dataset root path here
    DATASET_ROOT = "/home/datasets/personx"
    DATASET_NAME = "PersonX"

    def __init__(self, verbose=True, **kwargs):
        super(PersonX, self).__init__()

        # Absolute dataset directories
        self.dataset_dir = osp.join(self.DATASET_ROOT, self.DATASET_NAME)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        # Process dataset
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> PersonX loaded from: {}".format(self.dataset_dir))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check that all required directories exist"""
        required_dirs = [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir]
        for d in required_dirs:
            if not osp.exists(d):
                raise RuntimeError("'{}' is not available. Please check the folder.".format(d))

    def _process_dir(self, dir_path, relabel=False):
        """Process a directory and return list of (img_path, pid, camid)"""
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        cam2label = {3:1, 4:2, 8:3, 10:4, 11:5, 12:6}

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(osp.basename(img_path)).groups())
            pid_container.add(pid)
