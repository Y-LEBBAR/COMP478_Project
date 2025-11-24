import torch
import torch.nn as nn
import torch.nn.functional as F


class CosReLUSoftmaxLoss(nn.Module):
    """
    CosFace-style large margin cosine loss with ReLU-stabilized margin.
    Implements:
        L = -log( exp(s*(cos(theta_y)-m)) / sum_j exp(s*cos(theta_j)) )
    """
    def __init__(self, s=30.0, m=0.35):
        super().__init__()
        self.s = s
        self.m = m

    def forward(self, embeddings, labels, weights):
        # Normalize both embeddings and classifier weights
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(weights, dim=1)

        # Compute cosine similarities
        logits = F.linear(embeddings, weights)  # [B, C]

        # Subtract margin from target logits
        target_logits = logits.gather(1, labels.view(-1, 1))
        adjusted_target = torch.clamp(target_logits - self.m, min=-1.0)

        # Replace target positions with adjusted values
        logits = logits.scatter(1, labels.view(-1, 1), adjusted_target)


        # Scale and compute cross-entropy loss
        loss = F.cross_entropy(self.s * logits, labels)
        return loss
