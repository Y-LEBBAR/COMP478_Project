"""
model.py
--------
Defines CNN backbone and embedding projection for face recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class FaceNet(nn.Module):
    """
    Lightweight face recognition model based on ResNet18 backbone.
    Outputs L2-normalized embeddings and classification logits.
    """
    def __init__(self, num_classes: int = 7, emb_dim: int = 128):
        super().__init__()
        base = resnet18(weights="IMAGENET1K_V1")  # use pretrained ImageNet weights
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # remove final FC layer
        self.fc = nn.Linear(512, emb_dim)                           # embedding layer
        self.classifier = nn.Linear(emb_dim, num_classes)           # classification head

    def forward(self, x):
        """Forward pass through backbone → embedding → classification."""
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        emb = F.normalize(self.fc(x))        # normalize embeddings to unit sphere
        logits = self.classifier(emb)        # raw classification logits
        return emb, logits
