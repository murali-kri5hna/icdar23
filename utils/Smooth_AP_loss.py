import torch
from torch.cuda.amp import autocast
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f


"""
Disfunctional SmmothAP implementation prototype.
"""

def sigmoid(tensor, temp=1.0):
    """Temperature controlled sigmoid with memory optimizations"""
    # Directly clamp only where necessary to avoid intermediate tensor creation
    tensor = tensor / temp
    tensor = torch.where(tensor > 50, torch.tensor(50.0, device=tensor.device), tensor)
    tensor = torch.where(tensor < -50, torch.tensor(-50.0, device=tensor.device), tensor)
    y = torch.sigmoid(tensor)
    return y


class SmoothAP(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss."""

    def __init__(self, anneal=0.01):
        super(SmoothAP, self).__init__()

        self.anneal = anneal

    def forward(self, preds, labels, **kwargs):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """
        dtype, device = preds.dtype, preds.device

        N = labels.size(0)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels)
        I_pos = torch.zeros(N, N, dtype=dtype, device=device)
        I_neg = torch.zeros(N, N, dtype=dtype, device=device)
        I_pos[a1_idx, p_idx] = 1
        I_neg[a2_idx, n_idx] = 1

        sim_mat = torch.matmul(preds, preds.t())

        # Compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        # Compute the difference matrix and pass through the sigmoid
        
        sim_diff = sim_mat - sim_mat[:,0:1]
        
        sim_sg = sigmoid(sim_diff, temp=self.anneal)

        sim_neg = torch.sum(sim_sg * I_neg, dim=-1)

        sim_pos = torch.sum(sim_sg * I_pos, dim=-1)

        #breakpoint()
        
        ap = torch.sum((1 + sim_pos) / (1 + sim_pos + sim_neg))
        ap = ap/N
        
        return 1 - ap