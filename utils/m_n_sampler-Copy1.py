import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import numpy as np

class MWritersPerNClusterSampler(Sampler):
    """
    CustomSampler is a PyTorch Sampler that provides batches of data with specific constraints:
    - Each batch contains m samples per writer.
    - Within these m samples per writer, at least 2 different pages are represented.
    - Additionally, within these writer samples, there are n samples per cluster represented.
    
    Parameters:
    - clusters (array-like): An array of cluster identifiers for the samples.
    - writers (array-like): An array of writer identifiers for the samples.
    - pages (array-like): An array of page identifiers for the samples.
    - m (int): The number of samples per writer per batch.
    - n (int): The number of samples per cluster within each writer's samples.
    - batch_size (int): The size of each batch.
    - length_before_new_iter (int, optional): The length of the iteration before reshuffling the samples. Default is 100000.
    
    The sampler ensures:
    - Each writer in a batch has m samples.
    - There are at least 2 different pages represented in the m samples for each writer.
    - There are n samples per cluster for each writer within the selected samples.
    - The batch size is a multiple of m.
    
    Example Usage:
        clusters = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        writers = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3])
        pages = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
        m = 2  # samples per writer per batch
        n = 1  # samples per cluster within each writer
        batch_size = 6
        length_before_new_iter = 18
    
        sampler = CustomSampler(clusters, writers, pages, m, n, batch_size, length_before_new_iter)
        dataset = DummyDataset(np.arange(len(clusters)))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
        for batch in dataloader:
            print(batch)

    """
    def __init__(self, clusters, writers, pages, m, n, batch_size, length_before_new_iter=100000):
        if isinstance(clusters, torch.Tensor):
            clusters = clusters.numpy()
        if isinstance(writers, torch.Tensor):
            writers = writers.numpy()
        if isinstance(pages, torch.Tensor):
            pages = pages.numpy()
        
        self.m_per_writer = int(m)
        self.n_per_cluster = int(n)
        self.batch_size = int(batch_size)
        self.length_before_new_iter = length_before_new_iter

        # Organize data by writers, clusters, and pages
        self.writer_to_cluster_to_page_to_indices = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for idx, (cluster, writer, page) in enumerate(zip(clusters, writers, pages)):
            self.writer_to_cluster_to_page_to_indices[writer][cluster][page].append(idx)

        # Filter out writers that do not have at least n clusters
        self.writer_to_cluster_to_page_to_indices = {
            writer: cluster_dict
            for writer, cluster_dict in self.writer_to_cluster_to_page_to_indices.items()
            if len(cluster_dict) >= self.n_per_cluster
        }

        self.writers = list(self.writer_to_cluster_to_page_to_indices.keys())
        self.length_of_single_pass = self.m_per_writer * len(self.writers)
        self.list_size = length_before_new_iter

        assert self.list_size >= self.batch_size
        assert self.length_of_single_pass >= self.batch_size, "m * (number of unique writers) must be >= batch_size"
        assert self.batch_size % self.m_per_writer == 0, "batch_size must be a multiple of m_per_writer"
        self.list_size -= self.list_size % self.batch_size

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = []
        num_iters = self.calculate_num_iters()
        np.random.shuffle(self.writers)

        breakpoint()
        
        for _ in range(num_iters):
            curr_writer_set = self.writers[:self.batch_size // self.m_per_writer]
            for writer in curr_writer_set:
                clusters = list(self.writer_to_cluster_to_page_to_indices[writer].keys())
                np.random.shuffle(clusters)
                num_clusters = min(self.n_per_cluster, len(clusters))
                for cluster in clusters[:num_clusters]:
                    pages = list(self.writer_to_cluster_to_page_to_indices[writer][cluster].keys())
                    if len(pages) < 2:
                        continue
                    np.random.shuffle(pages)
                    selected_pages = pages[:2]
                    indices = []
                    for page in selected_pages:
                        indices.extend(self.writer_to_cluster_to_page_to_indices[writer][cluster][page])
                    selected_indices = np.random.choice(indices, size=min(self.m_per_writer, len(indices)), replace=False)
                    idx_list.extend(selected_indices)
        
        while len(idx_list) < self.list_size:
            idx_list.extend(idx_list[:self.list_size - len(idx_list)])

        breakpoint()

        return iter(idx_list[:self.list_size])

    def calculate_num_iters(self):
        divisor = self.batch_size
        return self.list_size // divisor if divisor < self.list_size else 1

if __name__ == "__main__":
    import numpy as np

    clusters = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    writers = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3])
    pages = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])

    m = 2  # samples per writer per batch
    n = 1  # samples per cluster within each writer
    batch_size = 6
    length_before_new_iter = 18

    sampler = CustomSampler(clusters, writers, pages, m, n, batch_size, length_before_new_iter)

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]

    dataset = DummyDataset(np.arange(len(clusters)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    for batch in dataloader:
        print(batch)

