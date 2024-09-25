import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses import BaseMetricLossFunction



class FastAPReward(BaseMetricLossFunction):
    def __init__(self, num_bins=10, **kwargs):
        super().__init__(**kwargs)
        c_f.assert_distance_type(self, LpDistance, normalize_embeddings=True, p=2)
        self.num_bins = int(num_bins)
        self.num_edges = self.num_bins + 1
        self.add_to_recordable_attributes(list_of_names=["num_bins"], is_stat=False)

    '''
    Adapted from FastAP loss implementation in the the FastAP-metric-learning repository
    '''

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        dtype, device = embeddings.dtype, embeddings.device
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        N = labels.size(0)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels)
        I_pos = torch.zeros(N, N, dtype=dtype, device=device)
        I_neg = torch.zeros(N, N, dtype=dtype, device=device)
        I_pos[a1_idx, p_idx] = 1
        I_neg[a2_idx, n_idx] = 1
        N_pos = torch.sum(I_pos, dim=1)
        safe_N = N_pos > 0
        
        N_pos_1 = N_pos - 1
        safe_N_1 = N_pos_1 > 0
        
        if torch.sum(safe_N) == 0:
            return self.zero_losses()
        dist_mat = self.distance(embeddings)

        histogram_max = 2**self.distance.power
        histogram_delta = histogram_max / self.num_bins
        mid_points = torch.linspace(
            0.0, histogram_max, steps=self.num_edges, device=device, dtype=dtype
        ).view(-1, 1, 1)
        pulse = torch.nn.functional.relu(
            1 - torch.abs(dist_mat - mid_points) / histogram_delta
        )
        pos_hist = torch.t(torch.sum(pulse * I_pos, dim=2))
        neg_hist = torch.t(torch.sum(pulse * I_neg, dim=2))

        first_I_pos, remaining_I_pos = split_matrix(I_pos)

        pos_hist_first = torch.t(torch.sum(pulse * first_I_pos, dim=2))
        pos_hist_baseline = torch.t(torch.sum(pulse * remaining_I_pos, dim=2))
        

        total_pos_hist = torch.cumsum(pos_hist, dim=1)
        total_hist = torch.cumsum(pos_hist + neg_hist, dim=1)

        h_pos_product = pos_hist * total_pos_hist
        safe_H = (h_pos_product > 0) & (total_hist > 0)
        if torch.sum(safe_H) > 0:
            FastAP = torch.zeros_like(pos_hist, device=device)
            FastAP[safe_H] = h_pos_product[safe_H] / total_hist[safe_H]

            FastAP_first, FastAP_baseline = split_matrix(FastAP)

            breakpoint()
            
            FastAP_baseline = torch.sum(FastAP_baseline, dim=1)[safe_N_1] / N_pos_1[safe_N_1]
            FastAP = torch.sum(FastAP_first, dim=1) - FastAP_baseline
            
            #FastAP = torch.sum(FastAP, dim=1)
            #FastAP = FastAP[safe_N] / N_pos[safe_N]
            FastAP = (1 - FastAP) * miner_weights[safe_N]
            return {
                "loss": {
                    "losses": FastAP,
                    "indices": torch.where(safe_N)[0],
                    "reduction_type": "element",
                }
            }
        return self.zero_losses()
    
    def compute_reward(self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None):
        c_f.check_shapes(embeddings, labels)
        if labels is not None:
            labels = c_f.to_device(labels, embeddings)
        ref_emb, ref_labels = c_f.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )
        return loss_dict["loss"]["losses"].detach().cpu().numpy()
    
    def get_default_distance(self):
        return LpDistance(power=2)


def split_matrix(matrix):
    n = matrix.size(0)
    first_positive_index_matrix = torch.zeros_like(matrix)
    remaining_positive_indexes_matrix = matrix.clone()
    
    for i in range(n):
        row = matrix[i]
        pos_indices = torch.nonzero(row > 0).squeeze()
        if pos_indices.numel() > 0:
            first_pos_index = pos_indices[0].item()
            first_positive_index_matrix[i, first_pos_index] = matrix[i, first_pos_index]
            remaining_positive_indexes_matrix[i, first_pos_index] = 0
    
    return first_positive_index_matrix, remaining_positive_indexes_matrix
