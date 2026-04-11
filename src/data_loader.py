import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class UTKFaceDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["path"]
        img_path = img_path.replace('\\', '/')
        age = self.dataframe.iloc[idx]["age"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(age, dtype=torch.float32)


def get_transforms(train=True):
    # Statistiques standards ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def get_data_loaders(csv_path, batch_size=32):
    df = pd.read_csv(csv_path)
    df["age_bin"] = pd.cut(df["age"], bins=10, labels=False)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["age_bin"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["age_bin"], random_state=42
    )

    train_ds = UTKFaceDataset(train_df, transform=get_transforms(train=True))
    val_ds = UTKFaceDataset(val_df, transform=get_transforms(train=False))
    test_ds = UTKFaceDataset(test_df, transform=get_transforms(train=False))

    # Création des DataLoaders (pour le batching et le multi-processing)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader
