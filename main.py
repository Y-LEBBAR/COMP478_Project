from datasets import load_dataset
import torch
from src.model import FaceNet
from src.train import train_one_epoch
from src.evaluate import evaluate

def main():
    # ------------------ Configuration ------------------
    cache_dir = "/content/drive/MyDrive/fairface_cache"
    num_classes = 7
    epochs = 10
    lr = 1e-4
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------ Load dataset from Drive ------------------
    ds = load_dataset("HuggingFaceM4/FairFace", "1.25", cache_dir=cache_dir)
    print("✅ Dataset loaded from Google Drive cache")

    from src.dataset import get_dataloaders
    train_loader, val_loader = get_dataloaders(use_hf=True, hf_dataset=ds, batch_size=batch_size)

    # ------------------ Model & Training ------------------
    model = FaceNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch:02d}] Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f}")

    torch.save(model.state_dict(), "cosface_fairface.pth")
    print("✅ Training complete. Model saved as cosface_fairface.pth")

if __name__ == "__main__":
    main()
