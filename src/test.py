import torch
import torch.nn as nn
from model import get_model
from data_loader import get_data_loaders
from tqdm import tqdm
import os


def test():
    csv_path = "data/processed/utkface_metadata.csv"
    model_path = "models/convnext_tiny_bs64_lr2.90e-04_wd1.02e-04_mae4.49.pth"
    batch_size = 64
    img_size = 224

    if not os.path.exists(model_path):
        print(f"Error: The file {model_path} was not found.")
        return
    _, _, test_loader = get_data_loaders(
        csv_path, batch_size=batch_size, img_size=img_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return
    model.to(device)
    model.eval()
    mae_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()

    total_mae = 0.0
    total_mse = 0.0
    num_samples = 0

    print(
        f"Starting inference on {len(test_loader.dataset)} test images using {device}..."
    )

    with torch.no_grad():
        for images, ages in tqdm(test_loader, desc="Testing"):
            images, ages = images.to(device), ages.to(device)
            outputs = model(images).view(-1)
            mae = mae_criterion(outputs, ages)
            mse = mse_criterion(outputs, ages)

            total_mae += mae.item() * images.size(0)
            total_mse += mse.item() * images.size(0)
            num_samples += images.size(0)

    final_mae = total_mae / num_samples
    final_rmse = (total_mse / num_samples) ** 0.5

    print("\nFINAL TEST RESULTS")
    print(f"Mean Absolute Error (MAE): {final_mae:.4f} years")
    print(f"Root Mean Squared Error (RMSE): {final_rmse:.4f} years")


if __name__ == "__main__":
    test()
