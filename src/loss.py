"""
loss.py
-------
Implements CosReLU Softmax loss — a simplified, stable variant of CosFace.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosReLUSoftmaxLoss(nn.Module):
    """
    Computes CosFace-style loss using ReLU-stabilized cosine margin.
    L = -log( exp(s*(cos(theta_y)-m)) / sum_j exp(s*cos(theta_j)) )
    """
    def __init__(self, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.s = s  # scaling factor
        self.m = m  # cosine margin

    def forward(self, embeddings, labels, weights):
        """
        Args:
            embeddings (Tensor): Normalized feature embeddings, shape [B, D]
            labels (Tensor): Ground-truth class indices, shape [B]
            weights (Tensor): Class weight matrix from model.classifier.weight
        """
        # Normalize classifier weights
        W = F.normalize(weights, dim=1)

        # Cosine similarities between embeddings and weights
        logits = F.linear(F.normalize(embeddings), W)  # [B, C]

        # Extract correct class cosine similarity
        target_cos = logits.gather(1, labels.view(-1, 1))

        # Apply ReLU-stabilized margin (prevents negative overshoot)
        adjusted_target = F.relu(target_cos - self.m)

        # Replace target logits with adjusted values
        scaled_logits = self.s * logits.clone()
        scaled_logits.scatter_(1, labels.view(-1, 1), self.s * adjusted_target)

        # Standard softmax cross-entropy
        loss = F.cross_entropy(scaled_logits, labels)
        return loss
