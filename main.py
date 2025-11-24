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
    data_root = "/content/drive/MyDrive/fairface_cache"  # Google Drive cache
    num_classes = 7       # races in FairFace dataset
    epochs = 10
    lr = 1e-4
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"✅ Using device: {device}")

    # ------------------ Data & Model -------------------
    print("📂 Loading dataset...")
    train_loader, val_loader = get_dataloaders(
        root_dir=data_root,
        batch_size=batch_size,
        use_hf=True   # use Hugging Face loader
    )

    print("✅ Dataset loaded successfully")

    model = FaceNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------ Training Loop ------------------
    print("🚀 Starting training...\n")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch:02d}] Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f}")

    # ------------------ Save Model ------------------
    torch.save(model.state_dict(), "cosface_fairface.pth")
    print("✅ Training complete. Model saved as cosface_fairface.pth")


if __name__ == "__main__":
    main()
