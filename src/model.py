import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FaceNet(nn.Module):
    """
    ResNet18-based feature extractor with cosine classification head.
    Outputs both embeddings (L2-normalized) and logits.
    """
    def __init__(self, num_classes=7, embedding_dim=512):
        super().__init__()

        # Backbone feature extractor
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # remove final fc
        in_features = base_model.fc.in_features

        # Embedding projection
        self.embedding_head = nn.Linear(in_features, embedding_dim)
        nn.init.xavier_uniform_(self.embedding_head.weight)

        # Cosine classifier
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        # Extract deep features
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        embedding = self.embedding_head(x)

        # Normalize embedding before classification
        embedding = F.normalize(embedding, dim=1)

        # Logits via cosine classifier
        logits = self.classifier(embedding)
        return embedding, logits
