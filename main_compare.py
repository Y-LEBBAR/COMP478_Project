"""
main_compare.py
---------------
Runs two experiments on FairFace-small (0.25):
1. CosReLU Softmax (our modified loss)
2. Regular Softmax (baseline)
Automatically saves models and compares overall + per-race accuracy.
"""

import torch
from src.dataset import get_dataloaders
from src.model import FaceNet
from src.train import train_one_epoch
from src.evaluate import evaluate
from src.loss import CosReLUSoftmaxLoss   # your existing version
from src.loss_baseline import SoftmaxLoss
import pandas as pd


def run_experiment(model_name, criterion_class, epochs=5, lr=1e-4, batch_size=64):
    """Train and evaluate one model configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Starting {model_name} training on device: {device}")

    # Data & model setup
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    model = FaceNet(num_classes=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = criterion_class()

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        overall_acc, group_acc = evaluate(model, val_loader, device)

        print(f"[{model_name}] Epoch {epoch}: Loss={train_loss:.4f}, Overall Acc={overall_acc:.3f}")
        for race, acc in group_acc.items():
            print(f"  {race}: {acc:.3f}")
        print("")

    # Save model
    torch.save(model.state_dict(), f"{model_name}_fairface_small.pth")
    print(f"✅ Saved {model_name}_fairface_small.pth")

    return overall_acc, group_acc


def compare_results(cos_results, softmax_results):
    """Prints a side-by-side performance comparison."""
    cos_overall, cos_groups = cos_results
    base_overall, base_groups = softmax_results

    print("\n📊 FINAL COMPARISON")
    print(f"Overall Accuracy → CosReLU: {cos_overall:.3f} | Softmax: {base_overall:.3f}\n")

    races = ["White","Black","Latino_Hispanic","East Asian",
             "Southeast Asian","Indian","Middle Eastern"]

    data = []
    for race in races:
        cos_acc = cos_groups.get(race, 0)
        base_acc = base_groups.get(race, 0)
        print(f"{race:<18} CosReLU={cos_acc:.3f} | Softmax={base_acc:.3f}")
        data.append({"Race": race, "CosReLU": cos_acc, "Softmax": base_acc})

    # Save comparison table for report
    df = pd.DataFrame(data)
    df.to_csv("fairface_small_comparison.csv", index=False)
    print("\n✅ Results saved to fairface_small_comparison.csv for report use.")


if __name__ == "__main__":
    # Run both models
    cosrelu_results = run_experiment("CosReLU", CosReLUSoftmaxLoss)
    softmax_results = run_experiment("Softmax", SoftmaxLoss)

    # Compare and export
    compare_results(cosrelu_results, softmax_results)
