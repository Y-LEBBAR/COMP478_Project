"""
main_compare.py
---------------
Runs two experiments on FairFace-small (0.25):
1. CosReLU Softmax (our modified loss)
2. Regular Softmax (baseline)
Automatically saves models and compares overall + per-race accuracy.
Also saves all results to Google Drive to prevent data loss.
"""

import torch
from src.dataset import get_dataloaders
from src.model import FaceNet
from src.train import train_one_epoch
from src.evaluate import evaluate
from src.loss import CosReLUSoftmaxLoss
from src.loss_baseline import SoftmaxLoss
import pandas as pd
import os


# Google Drive mount path (ensure Drive is mounted in Colab)
DRIVE_PATH = "/content/drive/MyDrive/FairFace_Results"
os.makedirs(DRIVE_PATH, exist_ok=True)


def run_experiment(model_name, criterion_class, epochs=5, lr=1e-4, batch_size=64):
    """
    Train and evaluate one model configuration (CosReLU or Softmax).
    Automatically saves model weights and metrics to Google Drive.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Starting {model_name} training on device: {device}")

    # Load FairFace-small (0.25) dataset from cached Drive folder
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    # Initialize model, optimizer, and loss
    model = FaceNet(num_classes=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = criterion_class()

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        overall_acc, group_acc = evaluate(model, val_loader, device)

        print(f"[{model_name}] Epoch {epoch}: Loss={train_loss:.4f}, Overall Acc={overall_acc:.3f}")
        for race, acc in group_acc.items():
            print(f"  {race}: {acc:.3f}")
        print("")

    # Save models locally and in Google Drive
    local_model_path = f"{model_name}_fairface_small.pth"
    drive_model_path = os.path.join(DRIVE_PATH, local_model_path)
    torch.save(model.state_dict(), local_model_path)
    torch.save(model.state_dict(), drive_model_path)
    print(f"✅ Saved model to {drive_model_path}")

    return overall_acc, group_acc


def compare_results(cos_results, softmax_results):
    """
    Compare per-race and overall accuracy between CosReLU and Softmax models.
    Exports results as CSV to both local and Google Drive.
    """
    cos_overall, cos_groups = cos_results
    base_overall, base_groups = softmax_results

    print("\n📊 FINAL COMPARISON")
    print(f"Overall Accuracy → CosReLU: {cos_overall:.3f} | Softmax: {base_overall:.3f}\n")

    races = [
        "White", "Black", "Latino_Hispanic", "East Asian",
        "Southeast Asian", "Indian", "Middle Eastern"
    ]

    data = []
    for race in races:
        cos_acc = cos_groups.get(race, 0)
        base_acc = base_groups.get(race, 0)
        print(f"{race:<18} CosReLU={cos_acc:.3f} | Softmax={base_acc:.3f}")
        data.append({"Race": race, "CosReLU": cos_acc, "Softmax": base_acc})

    df = pd.DataFrame(data)
    csv_name = "fairface_small_comparison.csv"
    csv_drive_path = os.path.join(DRIVE_PATH, csv_name)

    df.to_csv(csv_name, index=False)
    df.to_csv(csv_drive_path, index=False)
    print(f"\n✅ Results saved to {csv_drive_path} for report use.")


if __name__ == "__main__":
    # Cached dataset path on Google Drive
    data_root = "/content/drive/MyDrive/fairface_cache"

    # Run both experiments
    cosrelu_results = run_experiment("CosReLU", CosReLUSoftmaxLoss)
    softmax_results = run_experiment("Softmax", SoftmaxLoss)

    # Compare results and export
    compare_results(cosrelu_results, softmax_results)
