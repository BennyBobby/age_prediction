import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from src.data_loader import UTKFaceDataset, get_data_loaders


def _make_csv_and_images(tmp_path, n=10):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rows = []
    for i in range(n):
        path = img_dir / f"face_{i}.jpg"
        Image.new("RGB", (100, 100), color=(i * 20, 0, 0)).save(path)
        rows.append({"path": str(path), "age": float(20 + i)})
    csv_path = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(csv_path)


def test_dataset_length(tmp_path):
    csv = _make_csv_and_images(tmp_path, n=8)
    ds = UTKFaceDataset(csv)
    assert len(ds) == 8


def test_dataset_item_types(tmp_path):
    csv = _make_csv_and_images(tmp_path, n=4)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    ds = UTKFaceDataset(csv, transform=transform)
    img, age = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert isinstance(age, torch.Tensor)
    assert age.dtype == torch.float32


def test_dataset_age_values(tmp_path):
    csv = _make_csv_and_images(tmp_path, n=5)
    ds = UTKFaceDataset(csv)
    _, age = ds[0]
    assert age.item() == 20.0


def test_dataset_fallback_on_missing_image(tmp_path):
    # Row points to a non-existent file: dataset must not crash
    csv_path = tmp_path / "meta.csv"
    pd.DataFrame([{"path": "/nonexistent/face.jpg", "age": 30.0}]).to_csv(csv_path, index=False)
    ds = UTKFaceDataset(str(csv_path))
    img, age = ds[0]
    assert isinstance(img, Image.Image)
    assert age.item() == 30.0


def test_data_loaders_split_sizes(tmp_path):
    n = 20
    csv = _make_csv_and_images(tmp_path, n=n)
    train_loader, val_loader, test_loader = get_data_loaders(csv, batch_size=4)

    train_n = len(train_loader.dataset)
    val_n = len(val_loader.dataset)
    test_n = len(test_loader.dataset)

    assert train_n == int(0.8 * n)
    assert val_n == int(0.1 * n)
    assert test_n == n - train_n - val_n
    assert train_n + val_n + test_n == n
