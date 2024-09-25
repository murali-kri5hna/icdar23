import torch
from torch.cuda.amp import autocast
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f

def sigmoid(tensor, temp=0.01):
    """Temperature controlled sigmoid with memory optimizations"""
    # Directly clamp only where necessary to avoid intermediate tensor creation
    tensor = tensor / temp
    tensor = torch.where(tensor > 50, torch.tensor(50.0, device=tensor.device), tensor)
    tensor = torch.where(tensor < -50, torch.tensor(-50.0, device=tensor.device), tensor)
    return torch.sigmoid(tensor)


class SmoothAP(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss."""

    def __init__(self, anneal=0.01, **kwargs):
        super(SmoothAP, self).__init__()

        self.anneal = anneal

    def forward(self, preds, labels, **kwargs):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """
        dtype, device = preds.dtype, preds.device

        N = labels.size(0)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels)
        I_pos = torch.zeros(N, N, dtype=dtype, device=device) #+ torch.eye(N, dtype=dtype, device=device)
        I_neg = torch.zeros(N, N, dtype=dtype, device=device)# + torch.eye(N, dtype=dtype, device=device)
        I_pos[a1_idx, p_idx] = 1
        I_neg[a2_idx, n_idx] = 1

        sim_mat = torch.matmul(preds, preds.t())

        # Compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        # Compute the difference matrix and pass through the sigmoid

        sim_diff = sim_mat.unsqueeze(1) - sim_mat.unsqueeze(2)
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * (1 - torch.eye(N,device=device))

        
        sim_pos_sg = sim_sg * I_pos.unsqueeze(-1) * I_pos.unsqueeze(1)
        
        sim_pos_rk = torch.eye(N, device=device) + I_pos + torch.sum(sim_pos_sg, dim=-1)
        sim_all_rk = 1 + torch.sum(sim_sg, dim=-1)


        ap = torch.zeros(1).to(device) #.cuda()
        for i in range(N):
            N_pos = sum(I_pos[i,:]) + 1
            if N_pos - 1:
                ap += torch.sum(sim_pos_rk[i] / sim_all_rk[i]) / N_pos          
        
        ap = ap/N
        #breakpoint()
        
        return 1 - ap