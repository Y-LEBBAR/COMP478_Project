"""
test_run.py
------------
Quick sanity test to verify that FairFace dataloaders, GPU, and model setup work correctly.
"""

import torch
from dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a small batch to verify dataset and transformations
train_loader, val_loader = get_dataloaders(use_hf=True)

for i, (images, labels) in enumerate(train_loader):
    print(f"Batch {i+1} | Images shape: {images.shape} | Labels: {labels[:5]}")
    if i >= 3:  # only a few batches for quick check
        break
