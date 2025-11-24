""" dataset.py ---------- Defines FairFace dataset loading and augmentation pipelines. """
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image


def get_dataloaders(root_dir: str = None, batch_size: int = 64, img_size: int = 224, use_hf: bool = False):
    """
    Create train and validation dataloaders.
    If use_hf=True, load FairFace directly from Hugging Face (1.25 version).
    Otherwise, load from ImageFolder structure (root_dir/train, root_dir/val).
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    if use_hf:
        from torch.utils.data import Dataset

        class FairFaceHFDataset(Dataset):
            def __init__(self, split):
                self.ds = load_dataset("HuggingFaceM4/FairFace", "1.25", split=split)
                self.transform = transform

            def __len__(self): return len(self.ds)

            def __getitem__(self, idx):
                item = self.ds[idx]
                image = item["image"].convert("RGB")
                label = item["race"]
                return self.transform(image), label

        train_ds = FairFaceHFDataset("train")
        val_ds = FairFaceHFDataset("validation")

    else:
        train_ds = datasets.ImageFolder(os.path.join(root_dir, "train"), transform)
        val_ds   = datasets.ImageFolder(os.path.join(root_dir, "val"), transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader
