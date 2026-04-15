import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from model import get_model
from data_loader import get_data_loaders
import os


def train():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    accelerator = Accelerator(log_with="tensorboard", project_dir="logs")

    device = accelerator.device
    accelerator.print(f"Device used: {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        accelerator.print(f"GPU Name: {gpu_name}")
    else:
        accelerator.print("WARNING: CUDA not detected.")

    accelerator.init_trackers("age_estimation_convnext")
    epochs = 30
    lr = 0.0002900938030489172
    batch_size = 64
    weight_decay = 0.00010210191852411039
    model_type = "convnext_tiny"
    csv_path = "data/processed/utkface_metadata.csv"
    history = {"train_loss": [], "val_mae": []}
    train_loader, val_loader, _ = get_data_loaders(csv_path, batch_size=batch_size)
    model = get_model()
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    best_mae = float("inf")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        loop = tqdm(train_loader, disable=not accelerator.is_local_main_process)

        for images, ages in loop:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1), ages)
            accelerator.backward(loss)
            optimizer.step()

            total_train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        val_mae = 0
        with torch.no_grad():
            for images, ages in val_loader:
                outputs = model(images)
                loss = criterion(outputs.view(-1), ages)
                val_mae += loss.item()

        avg_val_mae = val_mae / len(val_loader)

        accelerator.log(
            {"train_loss": avg_train_loss, "val_mae": avg_val_mae}, step=epoch
        )

        history["train_loss"].append(avg_train_loss)
        history["val_mae"].append(avg_val_mae)

        accelerator.print(
            f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} | Val MAE {avg_val_mae:.4f}"
        )

        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            unwrapped_model = accelerator.unwrap_model(model)
            model_save_name = (
                f"convnext_tiny_"
                f"bs{batch_size}_"
                f"lr{lr:.2e}_"
                f"wd{weight_decay:.2e}_"
                f"mae{best_mae:.2f}.pth"
            )
            accelerator.save(unwrapped_model.state_dict(), f"models/{model_save_name}")
            accelerator.print(f"New best model saved: {model_save_name}")

    if accelerator.is_local_main_process:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training Loss ({model_type})")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history["val_mae"], label="Validation MAE", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("MAE (Years)")
        plt.title(f"Validation MAE ({model_type})")
        plt.legend()

        plot_name = (
            f"metrics_convnext_tiny_"
            f"bs{batch_size}_"
            f"lr{lr:.2e}_"
            f"wd{weight_decay:.2e}_"
            f"mae{best_mae:.2f}.png"
        )
        plt.savefig(f"reports/{plot_name}")
        accelerator.print(f"Metrics plot saved: {plot_name}")

    accelerator.end_training()


if __name__ == "__main__":
    train()
