import torch

from torch import nn
import torch.nn.functional as F

from backbone.resnets import resnet56, resnet20

class Model(torch.nn.Module):

    def __init__(self, backbone=resnet56(), num_clusters=100, dim=64, random=False):
        super(Model, self).__init__()
        self.backbone = backbone
        self.backbone.fc = torch.nn.Identity()
        self.nv = NetVLAD(num_clusters=num_clusters, dim=dim, random=random)
        self.classification = False

    def forward(self, x):
        if self.classification:
            emb = self.backbone(x)
            #breakpoint()
            #emb = emb.unsqueeze(-1).unsqueeze(-1)
            return F.normalize(emb)
            
        emb = self.backbone(x)                      # get residual features
        emb = emb.unsqueeze(-1).unsqueeze(-1)       # (NxD) -> (NxDx1x1)
        nv_enc = self.nv(emb)                       # encode features
        return F.normalize(nv_enc)                  # final normalization

class WriterResModel(torch.nn.Module):

    def __init__(self, backbone=resnet20(), num_clusters=100, dim=64, random=False):
        super(WriterResModel, self).__init__()
        self.backbone = backbone
        self.backbone.fc = torch.nn.Identity()
        self.nv = NetVLAD(num_clusters=num_clusters, dim=dim, random=random)
        self.classification = False

    def forward(self, x):
        if self.classification:
            x = x.unsqueeze(-1).unsqueeze(-1)
            emb = self.backbone(x)
            return F.normalize(emb)
            
        x = x.unsqueeze(-1).unsqueeze(-1)
        emb = self.backbone(x)                      # get residual features
        emb = emb.unsqueeze(-1).unsqueeze(-1)       # (NxD) -> (NxDx1x1)
        nv_enc = self.nv(emb)                       # encode features
     

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

        #vlad = F.normalize(vlad, p=2, dim=1)

        return vlad

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes=[6400, 394], final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)
        

class WriterModelFC(torch.nn.Module):
    def __init__(self, features=512, num_clusters=100, dim=64, random=False, dropout_prob=0.5):
        super(WriterModelFC, self).__init__()
        #self.fc = torch.nn.Linear(dim, num_writers)
        self.fc1 = nn.Linear(features, out_features=dim)  # Replace ... with the appropriate input feature size
        #self.fc2 = nn.Linear(in_features=1024, out_features=512)
        #self.fc3 = nn.Linear(in_features=512, out_features=dim)
        #self.dropout1 = nn.Dropout(p=dropout_prob)
        self.nv = NetVLAD(num_clusters=num_clusters, dim=dim, random=random)

        self._init_params()
        
    def _init_params(self):
        #self.fc1.weight.data.fill_(1.0) #0.01)
        torch.nn.init.eye_(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        #self.fc2.weight.data.normal_(0, 0.01)
        #self.fc2.bias.data.fill_(0)
       # 
    def forward(self, normalized_enc):
        #writer_enc = F.relu(self.fc1(normalized_enc))
        #writer_enc = self.dropout1(writer_enc)
        #writer_enc = self.fc2(writer_enc)
        #writer_enc = self.dropout1(writer_enc)
        writer_enc = self.fc1(normalized_enc)
        writer_enc = writer_enc.unsqueeze(-1).unsqueeze(-1)       # (NxD) -> (NxDx1x1)
        nv_writer_enc = self.nv(writer_enc)                       # encode features
        return F.normalize(nv_writer_enc) 
    
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
