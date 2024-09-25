import torch
from torch.cuda.amp import autocast
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f

def sigmoid(tensor, temp=0.01):
    """Temperature controlled sigmoid with memory optimizations"""
    tensor = tensor / temp
    tensor = torch.clamp(tensor, min=-50, max=50)
    return torch.sigmoid(tensor)

class SmoothAP(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss."""

    def __init__(self, anneal=0.01, **kwargs):
        super(SmoothAP, self).__init__()
        self.anneal = anneal

    def forward(self, preds, labels, **kwargs):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """
        torch.autograd.set_detect_anomaly(True)
        dtype, device = preds.dtype, preds.device
        N = labels.size(0)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels)
        
        I_pos = torch.zeros(N, N, dtype=dtype, device=device)
        I_pos[a1_idx, p_idx] = 1
        
        I_neg = torch.zeros(N, N, dtype=dtype, device=device)
        I_neg[a2_idx, n_idx] = 1

        # Compute the similarity matrix
        sim_mat = torch.matmul(preds, preds.t())

        # Compute the difference matrix and pass through the sigmoid
        sim_diff = sim_mat.unsqueeze(1) - sim_mat.unsqueeze(2)
        sim_sg = sigmoid(sim_diff, temp=self.anneal)
        sim_sg *= (1 - torch.eye(N, device=device))

        sim_pos_sg = sim_sg * I_pos.nsqueeze(-1) * I_pos.unsqueeze(1)

        # Compute the rankings
        sim_pos_rk = 1 + I_pos + torch.sum(sim_pos_sg, dim=-1)
        sim_all_rk = 1 + torch.sum(sim_sg, dim=-1)

        ap = torch.zeros(1, dtype=dtype, device=device)
        for i in range(N):
            N_pos = torch.sum(I_pos[i, :]) + 1
            if N_pos > 1:
                ap += torch.sum(sim_pos_rk[i] / sim_all_rk[i]) / N_pos

        ap /= N

        return 1 - ap

# Example usage:
if __name__ == "__main__":
    batch_size, feat_dims = 6, 256
    preds = torch.randn(batch_size, feat_dims, requires_grad=True).cuda()
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long).cuda()

    loss_fn = SmoothAP(anneal=0.01)
    loss = loss_fn(preds, labels)
    loss.backward()