import glob, os
import numpy as np
import shutil

import shutil
from multiprocessing.pool import ThreadPool


class MultithreadedCopier:
    def __init__(self, max_threads):
        self.pool = ThreadPool(max_threads)

    def copy(self, source, dest):
        self.pool.apply_async(shutil.copy2, args=(source, dest))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.close()
        self.pool.join()


pages = glob.glob('/cluster/qy41tewa/rl-map/dataset_patch/test/icdar2017_test_sift_patches_binarized_2kpp/*')

pages.sort()

#idxs = np.random.randint(1,3602,(600,))
idxs = np.arange(0,600)

with MultithreadedCopier(max_threads=64) as copier:
    for i in idxs:
        shutil.copytree(pages[i], f'/cluster/qy41tewa/rl-map/dataset_patch/debug/{os.path.basename(pages[i])}', copy_function=copier.copy, dirs_exist_ok = True)