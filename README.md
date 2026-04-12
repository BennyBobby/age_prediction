# Age Prediction with Deep Learning

This project implements a facial age estimation pipeline using PyTorch. It leverages transfer learning, distributed training with Hugging Face Accelerate, and automated hyperparameter optimization with Optuna.

## Key Features

- **Architecture:** ResNet50 backbone (Baseline)
- **Optimization:** Automated search for Learning Rate and Batch Size via Optuna
- **Distributed Training:** Configured with `accelerate` for high-performance training on RTX 5080
- **Monitoring:**
  - TensorBoard for real-time training curves (Loss/MAE)
  - Optuna Dashboard for hyperparameter importance analysis
- **Environment:** Managed with `uv` for fast dependency management

---

## Project Structure

```text
├── data/                   # Dataset and metadata (CSV)
├── logs/                   # TensorBoard event files
├── models/                 # Saved model checkpoints (.pth)
├── optuna/                 # SQLite database for optimization history
├── reports/                # Exported metrics plots (.png)
├── src/
│   ├── data_loader.py      # Custom Dataset and DataLoader
│   ├── model.py            # Model architecture 
│   ├── train.py            # Standard training script
│   └── train_optuna.py     # Hyperparameter tuning script
├── pyproject.toml          # Project dependencies
└── README.md
```

---

## Installation & Setup

1. Install `uv` (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Sync dependencies:

```bash
uv sync
```

3. Configure hardware:

```bash
uv run accelerate config
```

---

## How to Run

### 1. Hyperparameter Optimization

Run the Optuna study to find the best settings:

```bash
uv run python src/train_optuna.py
```

Visualize the study:

```bash
uv run optuna-dashboard sqlite:///optuna/optuna_study.db
```

### 2. Final Training

Train the model with your selected parameters:

```bash
uv run python src/train.py
```

### 3. Monitoring Progress

Launch TensorBoard to see live curves:

```bash
uv run tensorboard --logdir=logs --host=0.0.0.0 --port=6006
```