"""
train.py
--------
Implements one training epoch for CosFace variant.
"""

import torch
from tqdm import tqdm
from src.loss import CosReLUSoftmaxLoss


def train_one_epoch(model, loader, optimizer, device):
    """
    Train model for one epoch.
    """
    model.train()
    criterion = CosReLUSoftmaxLoss(s=30, m=0.35)
    running_loss = 0.0

    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        # Forward pass
        embeddings, _ = model(imgs)
        loss = criterion(embeddings, labels, model.classifier.weight)

        # Backward + optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)
