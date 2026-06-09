from __future__ import print_function, absolute_import
import os
import os.path as osp
import glob
import re
import sys

# 🔹 Add project root to Python path so imports work even when running directly
sys.path.append('/home/tq_sundas/IDM-main')

from idm.utils.data import BaseImageDataset
from idm.utils.osutils import mkdir_if_missing

class Market1501(BaseImageDataset):
    """
    Market1501 Dataset
    Reference:
    Zheng et al. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    # cameras: 6
    """

    # 🔹 Set your dataset absolute path here
    DATASET_ROOT = "/home/tq_sundas/dataset"
    DATASET_NAME = "market1501/Market-1501-v15.09.15"

    def __init__(self, verbose=True, **kwargs):
        super(Market1501, self).__init__()

        root = kwargs.get('root', osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))))
        self.dataset_dir = osp.join(root, 'Market-1501')
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        # Process dataset
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded from: {}".format(self.dataset_dir))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check that all required directories exist"""
        # Validate the main dataset folder
        expected_path = self.dataset_dir
        if not osp.exists(expected_path):
            raise RuntimeError("'{}' is not available. Please check the folder.".format(expected_path))
        
        required_dirs = [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir]
        for d in required_dirs:
            if not osp.exists(d):
                raise RuntimeError("'{}' is not available. Please check the folder.".format(d))

    def _process_dir(self, dir_path, relabel=False):
        """Process a directory and return list of (img_path, pid, camid)"""
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(osp.basename(img_path)).groups())
            if pid == -1:  # ignore junk images
                continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(osp.basename(img_path)).groups())
            if pid == -1:
                continue
            assert 0 <= pid <= 1501  # pid == 0 is background
            assert 1 <= camid <= 6
            camid -= 1  # zero-based indexing
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset


# 🔹 Test loading
if __name__ == "__main__":
    dataset = Market1501()
    print("Train samples:", len(dataset.train))
    print("Query samples:", len(dataset.query))
    print("Gallery samples:", len(dataset.gallery))
