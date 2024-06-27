import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size, margin=0.5, scale=64):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize the feature vectors and the weight vectors
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute the cosine similarity
        cos_theta = torch.matmul(embeddings, weight.t())
        cos_theta = cos_theta.clamp(-1, 1)  # Ensure cosine values are in range
        
        # Add the margin to the cosine similarity of the true class
        target_logit = cos_theta[torch.arange(0, embeddings.size(0)), labels]
        target_logit = torch.cos(torch.acos(target_logit) + self.margin)
        
        # Scale the logits
        logits = self.scale * cos_theta
        logits[torch.arange(0, embeddings.size(0)), labels] = self.scale * target_logit
        
        # Compute the loss
        loss = F.cross_entropy(logits, labels)
        return loss


if __name__ == '__main__':
    # Example usage:
    # Define the model and loss
    class ExampleModel(nn.Module):
        def __init__(self, embedding_size, num_classes):
            super(ExampleModel, self).__init__()
            self.backbone = nn.Sequential(
                # Add your backbone layers here
                nn.Linear(512, embedding_size)
            )
            self.arcface_loss = ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size)

        def forward(self, x, labels=None):
            embeddings = self.backbone(x)
            if labels is not None:
                loss = self.arcface_loss(embeddings, labels)
                return embeddings, loss
            return embeddings

    # Create model and optimizer
    embedding_size = 128
    num_classes = 10
    model = ExampleModel(embedding_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for data, labels in dataloader:
        optimizer.zero_grad()
        embeddings, loss = model(data, labels)
        loss.backward()
        optimizer.step()
