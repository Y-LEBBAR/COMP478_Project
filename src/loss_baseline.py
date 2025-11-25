import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxLoss(nn.Module):
    """Standard cross-entropy loss for baseline comparison."""
    def forward(self, embeddings, labels, weights):
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(weights, dim=1)
        logits = F.linear(embeddings, weights)
        return F.cross_entropy(logits, labels)
