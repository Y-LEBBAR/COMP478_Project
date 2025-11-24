import torch
from tqdm import tqdm
from src.loss import CosReLUSoftmaxLoss


def gradient_check(model, loader, device):
    """
    One-time gradient flow diagnostic.
    Run this once before full training to confirm gradients reach all parameters.
    """
    imgs, labels = next(iter(loader))
    imgs, labels = imgs.to(device), labels.to(device)

    criterion = CosReLUSoftmaxLoss(s=30, m=0.35)
    emb, _ = model(imgs)
    loss = criterion(emb, labels, model.classifier.weight)
    loss.backward()

    print("\n--- Gradient Check ---")
    for name, param in model.named_parameters():
        print(f"{name:40s} → grad: {param.grad is not None}")
    print("-----------------------\n")

    # stop here if running this as a diagnostic
    # exit()  # uncomment for one-time test


def train_one_epoch(model, loader, optimizer, device, epoch=None):
    """
    Train model for one epoch using CosReLU Softmax Loss, with gradient diagnostics.
    """
    model.train()
    criterion = CosReLUSoftmaxLoss(s=30, m=0.35)
    running_loss = 0.0

    for imgs, labels in tqdm(loader, desc=f"Training Epoch {epoch}", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        # Forward pass
        embeddings, logits = model(imgs)

        # Compute loss
        loss = criterion(embeddings, labels, model.classifier.weight)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Diagnostic: check classifier gradient norm
        #with torch.no_grad():
        #    grad_norm = model.classifier.weight.grad.norm().item()
        #print(f"Classifier weight grad norm: {grad_norm:.6f}")

        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    # Monitor classifier weight norm at end of epoch
    with torch.no_grad():
        w_norm = model.classifier.weight.norm(dim=1).mean().item()
    print(f"[Epoch {epoch}] Avg classifier weight norm: {w_norm:.3f}")

    return avg_loss
