"""
evaluate.py
------------
Evaluation utilities for accuracy computation and visualization.
"""

import torch
from sklearn.metrics import accuracy_score


def evaluate(model, loader, device):
    """
    Evaluate model accuracy on validation set.
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, logits = model(imgs)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    return acc
