FairFace CosFace/CosReLU vs Softmax – Project README
COMP 478 – Deep Learning • Face Recognition Fairness Study
Author: Yannis Lebbar
# 1. Project Overview

This project investigates whether CosFace-style angular-margin learning improves fairness and classification performance on small, balanced facial datasets, specifically FairFace 0.25 and FairFace 1.25.

Two models are compared:

CosReLU Softmax Loss

A stabilized variant of CosFace

Normalizes feature embeddings and classifier weights

Applies a cosine margin with hyperparameters 
m=0.35 
s=30

Prevents negative overshoot via ReLU-style clamping

Standard Softmax Cross-Entropy

Baseline classifier with normalized weights

No angular margin

Both models use a ResNet-18 backbone and a 512-dimensional embedding head.

The goal is not to match CosFace’s 99%+ benchmark accuracies (which require millions of images), but to evaluate whether angular margins help under small-data, fairness-focused conditions.

# 2. Repository Structure
COMP478_Project/
│
├── data/
│   ├── 478 Results - COSRELU.csv
│   ├── 478 Results - Softmax.csv
│   ├── fairface/                   # FairFace dataset (0.25 or 1.25)
│   ├── synthetic_augmented/        # Optional synthetic expansions
│   ├── cosrelu_vs_softmax.png      # Comparison plots
│   ├── ff125_loss.png              # FairFace 1.25 training loss
│   └── ff125_val_acc.png           # FairFace 1.25 validation accuracy
│
├── src/
│   ├── dataset.py                  # Dataset loader + HuggingFace support
│   ├── model.py                    # ResNet-18 + embedding + cosine classifier
│   ├── train.py                    # Training loop + gradient flow check
│   ├── evaluate.py                 # Per-race accuracy evaluation
│   ├── loss.py                     # CosReLU Softmax (CosFace variant)
│   ├── loss_baseline.py            # Standard Softmax loss
│   ├── utils.py                    # Helper functions (misc.)
│   └── test_run.py                 # Quick dataset/model sanity test
│
├── main_compare.py                 # Full training + comparison pipeline
├── main.py                         # Basic single-model training (optional)
├── requirements.txt
├── README.md
└── .gitignore

# 3. Installation & Environment Setup
Python version

Python 3.9+ is recommended.

Install dependencies
pip install -r requirements.txt

GPU Recommended

Training on CPU is possible but extremely slow.
Google Colab GPU (T4) was used for all experiments.

# 4. Dataset Setup – FairFace
Option A — Local Folder

Place FairFace here:

data/fairface/


with subfolders:

train/
val/

Option B — HuggingFace (Recommended)

dataset.py supports loading FairFace directly from HuggingFace.

train_loader, val_loader = get_dataloaders(use_hf=True)

# 5. Running the Project
5.1 Quick Sanity Test

This verifies:

Dataset loads successfully

GPU is recognized

Model forward pass works

Run:

python src/test_run.py


(See file: test_run)

5.2 Full Experiment Comparison

Runs CosReLU and Softmax, saves:

trained models

CSV results

per-race accuracy

comparison plots

Run:

python main_compare.py


This executes:

10 epochs CosReLU

10 epochs Softmax

Exports CSVs to /data

Generates comparative graphs

# 6. Model Architecture
Backbone

ResNet-18 (18 Layer CNN)

Pretrained on ImageNet

Modified to output 512-dimensional embeddings



after both are normalized:

embeddings = F.normalize(embeddings, dim=1)
weights = F.normalize(weights, dim=1)
logits = F.linear(embeddings, weights)

# 7. Loss Functions
7.1 CosReLU Softmax (CosFace Variant)

(See file: src/loss.py)

Key logic:

target_logits = logits.gather(1, labels.view(-1, 1))
adjusted_target = torch.clamp(target_logits - m, min=-1.0)
logits = logits.scatter(1, labels.view(-1, 1), adjusted_target)
loss = F.cross_entropy(s * logits, labels)


Applies margin 
m=0.35

Clamps to avoid negative overshoot

Multiplies logits by 
𝑠 =30 for sharper decision boundaries

7.2 Standard Softmax (Baseline)

(See file tagged: train.py)

logits = F.linear(embeddings, weights)
loss = F.cross_entropy(logits, labels)


No angular margin, no clamping.

# 8. Evaluation
8.1 Per-Race Accuracy

The evaluator computes:

White
Black
East Asian
Southeast Asian
Indian
Middle Eastern
Latino/Hispanic


(See: src/evaluate.py)

8.2 Exported CSV Files

478 Results - COSRELU.csv

478 Results - Softmax.csv

These track:

Overall accuracy per epoch

Per-race accuracy per epoch

# 9. Experiments & Results
9.1 FairFace 0.25 (Small dataset)

Final accuracy (epoch 10):

Model	Final Accuracy
CosReLU	0.652
Softmax	0.655
Observation

CosReLU does not outperform Softmax under small-data constraints.

BUT CosReLU produces more stable per-race training curves.

9.2 FairFace 1.25 (Larger dataset)

Extracted from training logs:

Epoch	Loss	Validation Accuracy
1	8.689	0.618
2	7.236	0.643
3	6.546	0.652
4	5.930	0.680
5	5.353	0.678
6	4.832	0.679
7	4.305	0.674
8	3.841	0.679
9	3.400	0.664
10	3.044	0.677

Key takeaway:
→ Larger, more diverse data gave ~2–3% accuracy increase,
confirming CosFace benefits more from abundance of data.

# 10. Visualizations

This repo includes:

cosrelu_vs_softmax_validation_accuracy.png

epoch_loss.png

ff125_val_acc.png

ff125_loss.png

Seven race-specific accuracy plots (white.png, black.png, etc.)

These were generated via data_visualization.py.

# 11. Reproducibility & How to Run

To fully reproduce:

python main_compare.py


To run full FairFace 1.25 training:

python main.py 

# 12. Requirements

All required packages are listed in:

requirements.txt


This includes:

PyTorch

torchvision

pandas

matplotlib

tqdm

huggingface datasets