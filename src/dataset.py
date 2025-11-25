import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset

class FairFaceHFDataset(Dataset):
    def __init__(self, split="train", img_size=224, version="0.25"):
        self.ds = load_dataset("HuggingFaceM4/FairFace", version, split=split)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        label = item["race"]
        return self.transform(image), label


def get_dataloaders(batch_size=64, img_size=224):
    train_ds = FairFaceHFDataset(split="train", img_size=img_size, version="0.25")
    val_ds = FairFaceHFDataset(split="validation", img_size=img_size, version="0.25")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader
