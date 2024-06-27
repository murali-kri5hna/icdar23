import random
from collections import defaultdict
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.samplers import MPerClassSampler

class MNSampler(MPerClassSampler):
    def __init__(self, labels, m, n, batch_size, writer_labels, page_labels, **kwargs):
        super().__init__(labels, m, batch_size, **kwargs)
        self.writer_labels = writer_labels
        self.page_labels = page_labels
        self.n = n
        self.m = m

        # Group indices by writer
        self.writer_to_indices = defaultdict(list)
        for idx, writer in enumerate(writer_labels):
            self.writer_to_indices[writer].append(idx)

        # Group indices by page
        self.page_to_indices = defaultdict(list)
        for idx, page in enumerate(page_labels):
            self.page_to_indices[page].append(idx)

    def __iter__(self):
        indices = super().__iter__()
        sampled_indices = list(indices)

        # Ensure at least n pages per author
        writer_count = defaultdict(int)
        writer_page_set = defaultdict(set)
        for idx in sampled_indices:
            writer = self.writer_labels[idx]
            page = self.page_labels[idx]
            writer_page_set[writer].add(page)
        
        for writer, pages in writer_page_set.items():
            if len(pages) < self.n:
                needed_pages = self.n - len(pages)
                available_pages = [p for p in self.page_to_indices if p not in pages]
                additional_pages = random.sample(available_pages, min(needed_pages, len(available_pages)))
                
                for page in additional_pages:
                    page_indices = self.page_to_indices[page]
                    sampled_indices.extend(page_indices[:self.m])
                    
        # Ensure m elements per cluster
        final_indices = []
        clusters = defaultdict(list)
        for idx in sampled_indices:
            cluster = self.labels[idx]
            clusters[cluster].append(idx)
        
        for cluster, cluster_indices in clusters.items():
            if len(cluster_indices) >= self.m:
                final_indices.extend(random.sample(cluster_indices, self.m))
            else:
                final_indices.extend(cluster_indices)
        
        random.shuffle(final_indices)
        return iter(final_indices)

    def __len__(self):
        return len(self.labels)

# Example usage
# labels = [...]  # Your cluster labels
# writer_labels = [...]  # Your writer labels
# page_labels = [...]  # Your page labels
# m = 5  # Number of elements per cluster
# n = 2  # Minimum number of pages per author
# batch_size = 32  # Desired batch size

# custom_sampler = CustomSampler(labels, m, n, batch_size, writer_labels, page_labels)

# # Now you can use the custom sampler with a DataLoader
# from torch.utils.data import DataLoader

# data_loader = DataLoader(your_dataset, batch_size=batch_size, sampler=custom_sampler)