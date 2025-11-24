"""
main.py
-------
Entry point for training and evaluating CosFace variant on FairFace dataset.
"""

import torch
from src.dataset import get_dataloaders
from src.model import FaceNet
from src.train import train_one_epoch
from src.evaluate import evaluate


def main():
    # ------------------ Configuration ------------------
    data_root = "data/fairface"
    num_classes = 7       # races in FairFace dataset
    epochs = 10
    lr = 1e-4
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------ Data & Model -------------------
    train_loader, val_loader = get_dataloaders(data_root, batch_size)
    model = FaceNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------ Training Loop ------------------
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch:02d}] Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f}")

    # Save trained weights
    torch.save(model.state_dict(), "cosface_fairface.pth")
    print("✅ Training complete. Model saved as cosface_fairface.pth")


if __name__ == "__main__":
    main()
