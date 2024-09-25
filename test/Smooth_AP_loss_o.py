import torch

def sigmoid(tensor, temp=1.0):
    """Temperature controlled sigmoid

    Takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # Clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y

def compute_aff(x):
    """Computes the affinity matrix between an input vector and itself"""
    return torch.mm(x, x.t())

class SmoothAP(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.

    Implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.

    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.

    Args:
        anneal : float
            The temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the Heaviside step function in the ranking function.
        batch_size : int
            The batch size being used during training.
        num_id : int
            The number of different classes that are represented in the batch.
        feat_dims : int
            The dimension of the input feature embeddings.

    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    """

    def __init__(self, anneal, batch_size, num_id, feat_dims):
        super(SmoothAP, self).__init__()

        assert batch_size % num_id == 0

        self.anneal = anneal
        self.batch_size = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims

    def forward(self, preds, **kwargs):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """

        # ------ differentiable ranking of all retrieval set ------
        # Compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(self.batch_size, device=preds.device)
        # Compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        sim_all = compute_aff(preds)
        # Compute the difference matrix and pass through the sigmoid
        sim_diff = sim_all.unsqueeze(1) - sim_all.unsqueeze(2)
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask
        # Compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # Compute the mask which only gives non-zero weights to the positive set
        xs = preds.view(self.num_id, self.batch_size // self.num_id, self.feat_dims)
        pos_mask = 1.0 - torch.eye(self.batch_size // self.num_id, device=preds.device)
        # Compute the relevance scores
        sim_pos = torch.bmm(xs, xs.permute(0, 2, 1))
        # Compute the difference matrix and pass through the sigmoid
        sim_pos_diff = sim_pos.unsqueeze(2) - sim_pos.unsqueeze(3)
        sim_pos_sg = sigmoid(sim_pos_diff, temp=self.anneal) * pos_mask
        # Compute the rankings of the positive set
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1

        # Sum the values of the Smooth-AP for all instances in the mini-batch
        ap = torch.zeros(1, device=preds.device)
        group = self.batch_size // self.num_id
        for ind in range(self.num_id):
            sim_all_rk_slice = sim_all_rk[ind * group:(ind + 1) * group, ind * group:(ind + 1) * group]
            pos_divide = torch.sum(sim_pos_rk[ind] / sim_all_rk_slice)
            ap += (pos_divide / group) / self.batch_size

        return 1 - ap