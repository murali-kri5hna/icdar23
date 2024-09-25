import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import numpy as np

class NPerClusterSampler(Sampler):
    """
    At every iteration, this will return m samples per cluster, and within each cluster,
    there will be n samples per writer.
    """

    def __init__(self, clusters, writers, pages, m, n, batch_size=None, length_before_new_iter=100000):
        if isinstance(clusters, torch.Tensor):
            clusters = clusters.numpy()
        if isinstance(writers, torch.Tensor):
            writers = writers.numpy()
            
        self.m_per_cluster = int(m)
        self.n_per_writer = int(n)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size
        self.length_before_new_iter = length_before_new_iter

        self.cluster_to_writer_to_indices = defaultdict(lambda: defaultdict(list))
        for idx, (cluster, writer) in enumerate(zip(clusters, writers)):
            self.cluster_to_writer_to_indices[cluster][writer].append(idx)

        # Filter out clusters that do not have at least n writers
        self.cluster_to_writer_to_indices = {
            cluster: writer_dict
            for cluster, writer_dict in self.cluster_to_writer_to_indices.items()
            if len(writer_dict) >= self.n_per_writer
        }

        self.clusters = list(self.cluster_to_writer_to_indices.keys())
        self.length_of_single_pass = self.m_per_cluster * len(self.clusters)
        self.list_size = length_before_new_iter
        
        if self.batch_size is None:
            if self.length_of_single_pass < self.list_size:
                self.list_size -= self.list_size % self.length_of_single_pass
        else:
            assert self.list_size >= self.batch_size
            assert (
                self.length_of_single_pass >= self.batch_size
            ), "m * (number of unique clusters) must be >= batch_size"
            assert (
                self.batch_size % self.m_per_cluster
            ) == 0, "batch_size must be a multiple of m_per_cluster"
            self.list_size -= self.list_size % self.batch_size

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = []
        num_iters = self.calculate_num_iters()
        np.random.shuffle(self.clusters)
        for _ in range(num_iters):
            if self.batch_size is None:
                curr_cluster_set = self.clusters
            else:
                curr_cluster_set = self.clusters[: self.batch_size // self.m_per_cluster]
            for cluster in curr_cluster_set:
                writers = list(self.cluster_to_writer_to_indices[cluster].keys())
                np.random.shuffle(writers)
                num_writers = min(self.m_per_cluster, len(writers))
                for writer in writers[:num_writers]:
                    indices = self.cluster_to_writer_to_indices[cluster][writer]
                    selected_indices = np.random.choice(indices, size=min(self.n_per_writer, len(indices)), replace=False)
                    idx_list.extend(selected_indices)
        
        # If we have fewer indices than required, repeat until we have enough
        while len(idx_list) < self.list_size:
            idx_list.extend(idx_list[:self.list_size - len(idx_list)])

        return iter(idx_list[:self.list_size])

    def calculate_num_iters(self):
        divisor = (
            self.length_of_single_pass if self.batch_size is None else self.batch_size
        )
        return self.list_size // divisor if divisor < self.list_size else 1

if __name__ == "__main__":
    import numpy as np

    # Example data
    clusters = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    writers = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3])
    ü = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
    labels = np.array(list(zip(clusters, writers)))

    # Parameters
    m = 2  # samples per cluster
    n = 1  # samples per writer within each cluster
    batch_size = 6
    length_before_new_iter = 18

    # Instantiate sampler
    sampler = NPerClusterSampler(clusters, writers, m, n, batch_size, length_before_new_iter)
    
    # Create a dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]

    dataset = DummyDataset(np.arange(len(clusters)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Testing the sampler
    for batch in dataloader:
        print(batch)