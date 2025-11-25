"""
evaluate.py
------------
Evaluation utilities for accuracy computation and visualization.
"""

import torch
from sklearn.metrics import accuracy_score
from collections import defaultdict

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_races = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            embeddings, logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # if your dataset returns race labels separately, append them here
            # otherwise, you can map numeric race IDs to race names later
            all_races.extend(labels.cpu().numpy())

    overall_acc = accuracy_score(all_labels, all_preds)

    # Per-race accuracy
    group_acc = defaultdict(list)
    for label, pred, race in zip(all_labels, all_preds, all_races):
        group_acc[race].append(pred == label)

    group_acc = {race: sum(accs)/len(accs) for race, accs in group_acc.items()}

    return overall_acc, group_acc

