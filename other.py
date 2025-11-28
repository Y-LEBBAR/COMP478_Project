"""
plots.py
---------------
Used to generate various plots that were used for the final report
"""
import matplotlib.pyplot as plt

# Shared epochs
epochs = list(range(1, 11))

# FairFace 1.25 metrics
val_acc = [0.618, 0.643, 0.652, 0.680, 0.678,
           0.679, 0.674, 0.679, 0.664, 0.677]

losses = [8.6895, 7.2368, 6.5468, 5.9302, 5.3536,
          4.8327, 4.3055, 3.8415, 3.4008, 3.0440]

# 1) Validation Accuracy plot
plt.figure(figsize=(6, 4))
plt.plot(epochs, val_acc, marker='o', linewidth=2)
plt.title("Validation Accuracy over Epochs (FairFace 1.25)")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.grid(True, alpha=0.3)
plt.xticks(epochs)
plt.tight_layout()
plt.savefig("ff125_val_acc.png", dpi=300)
plt.close()

# 2) Training Loss plot
plt.figure(figsize=(6, 4))
plt.plot(epochs, losses, marker='o', linewidth=2)
plt.title("Training Loss over Epochs (FairFace 1.25)")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.grid(True, alpha=0.3)
plt.xticks(epochs)
plt.tight_layout()
plt.savefig("ff125_loss.png", dpi=300)
plt.close()
