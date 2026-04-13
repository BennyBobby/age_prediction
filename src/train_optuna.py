import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm.auto import tqdm
import optuna
from model import get_model
from data_loader import get_data_loaders
import os


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    os.makedirs("models", exist_ok=True)
    accelerator = Accelerator()

    epochs = 10
    csv_path = "data/processed/utkface_metadata.csv"

    train_loader, val_loader, _ = get_data_loaders(csv_path, batch_size=batch_size)

    model = get_model()
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    best_trial_mae = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        loop = tqdm(
            train_loader,
            desc=f"Trial {trial.number} - Epoch {epoch+1}",
            disable=not accelerator.is_local_main_process,
        )

        for images, ages in loop:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1), ages)
            accelerator.backward(loss)
            optimizer.step()
            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for images, ages in val_loader:
                outputs = model(images)
                loss = criterion(outputs.view(-1), ages)
                val_mae += loss.item()

        avg_val_mae = val_mae / len(val_loader)

        trial.report(avg_val_mae, epoch)

        # If the trial is performing poorly compared to others, stop it early
        if trial.should_prune():
            accelerator.end_training()
            raise optuna.exceptions.TrialPruned()

        if avg_val_mae < best_trial_mae:
            best_trial_mae = avg_val_mae

    accelerator.end_training()
    return best_trial_mae


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        study_name="age_estimation_optimization_convnext-tiny",
        storage="sqlite:///optuna/optuna_study.db",
        load_if_exists=True,
    )

    print("Starting Optuna optimization...")
    study.optimize(objective, n_trials=15)

    print("\n" + "=" * 30)
    print("OPTIMIZATION COMPLETED")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best MAE: {study.best_value:.2f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print("=" * 30)
