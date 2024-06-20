import torch

from torch import nn
import torch.nn.functional as F

from backbone.resnets import resnet56
import argparse

class Model(torch.nn.Module):

    def __init__(self, backbone=resnet56(), num_clusters=100, dim=64, random=False):
        super(Model, self).__init__()
        self.backbone = backbone
        # “The last fully connected layer of the network is dropped, and the output of the global 
        # averaging pooling layer of dimension (64, 1, 1) is used” 
        # ([Peer et al., 2023, p. 5](zotero://select/library/items/73PPT7VX)) 
        # ([pdf](zotero://open-pdf/library/items/LCJPH7KQ?page=5&annotation=IMN2HKB6))
        self.backbone.fc = torch.nn.Identity()
        self.nv = NetVLAD(num_clusters=num_clusters, dim=dim, random=random)

    def forward(self, x):
        emb = self.backbone(x)                      # get residual features
        emb = emb.unsqueeze(-1).unsqueeze(-1)       # (NxD) -> (NxDx1x1)
        nv_enc = self.nv(emb)                       # encode features
        return F.normalize(nv_enc)                  # final normalization

class NetVLAD(nn.Module):
    """Net(R)VLAD layer implementation"""

    def __init__(self, num_clusters=100, dim=64, alpha=100.0, random=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            random : bool
                enables NetRVLAD, removes alpha-init and normalization

        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.random = random
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)

        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        if not self.random:
            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C = x.shape[:2]

        if not self.random:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        
        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        # x = self.pool(x)
        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        
        residual *= soft_assign.unsqueeze(2)

        vlad = residual.sum(dim=-1)

        if not self.random:
            vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        
        vlad = vlad.view(x.size(0), -1)  # flatten

        if not self.random:
            vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
    
class RewardtuneModelFC(torch.nn.Module):
    def __init__(self, dim=64, num_writers=394):
        self.fc = torch.nn.Linear(dim, num_writers)
        
    def forward(self, normalized_nv_enc):
        output = self.fc(normalized_nv_enc)
        softmax_output = F.softmax(output, dim=1)
        return softmax_output
    
class RewardtuneModel(torch.nn.Module):
    def __init__(self, model, rewardtune_model_fc):
        super(RewardtuneModel, self).__init__()
        self.model = model
        self.rewardtune_model_fc = rewardtune_model_fc

    def forward(self, x):
        embeddings = self.model(x)
        output = self.rewardtune_model_fc(embeddings)
        return output




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_finetune', default=False, action='store_true',
                        help='only test')

    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth',
                        help='path to the checkpoint file')
    args = parser.parse_args()

    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint_path)

    # Create an instance of the Model class
    model = Model()
    
    # Freeze the parameters of the backbone network
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Freeze the parameters of the NetVLAD network
    for param in model.nv.parameters():
        param.requires_grad = False

    # Load the weights from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()
    

    # Create an instance of the RewardtuneModelFC class
    rewardtune_model_fc = RewardtuneModelFC()

    # Train only the fully connected layer
    optimizer = torch.optim.Adam(rewardtune_model_fc.fc.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()

            # Pass the images through the pretrained model to get the embeddings
            with torch.no_grad():
                embeddings = model(images)

            outputs = rewardtune_model_fc(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Unfreeze the parameters of the backbone network
    for param in model.backbone.parameters():
        param.requires_grad = True
    
    # Unfreeze the parameters of the NetVLAD network
    for param in model.nv.parameters():
        param.requires_grad = True

    # Train the full network
    optimizer = torch.optim.Adam(rewardtune_model_fc.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()

            # Pass the images through the pretrained model to get the embeddings
            with torch.no_grad():
                embeddings = model(images)

            outputs = rewardtune_model_fc(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
