"""
main.py
-------
Entry point for training and evaluating CosFace variant on FairFace dataset.
"""

import torch
from src.dataset import get_dataloaders
from src.model import FaceNet
from src.train import train_one_epoch, gradient_check
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

    # ------------------ Gradient Flow Check (One-Time) ------------------
    print("\n🔍 Running one-time gradient flow check...")
    gradient_check(model, train_loader, device)
    print("✅ Gradient check complete. If all params show True, proceed.\n")

    # ------------------ Training Loop ------------------
    print("🚀 Starting training...\n")
    for epoch in range(1, epochs + 1):
        # Track classifier weight norm before training epoch
        with torch.no_grad():
            w_norm_start = model.classifier.weight.norm(dim=1).mean().item()
        print(f"[Epoch {epoch}] Start classifier weight norm: {w_norm_start:.3f}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_acc = evaluate(model, val_loader, device)

        # Track classifier weight norm after training epoch
        with torch.no_grad():
            w_norm_end = model.classifier.weight.norm(dim=1).mean().item()
        print(f"[Epoch {epoch}] End classifier weight norm: {w_norm_end:.3f}")
        print(f"[Epoch {epoch:02d}] Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f}\n")

    # ------------------ Save Model ------------------
    torch.save(model.state_dict(), "cosface_fairface.pth")
    print("✅ Training complete. Model saved as cosface_fairface.pth")


if __name__ == "__main__":
    main()
