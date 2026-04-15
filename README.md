# Age Prediction with Deep Learning

This project implements a facial age estimation pipeline using PyTorch. It leverages transfer learning, distributed training with Hugging Face Accelerate, and automated hyperparameter optimization with Optuna.

## Key Features

- **Architecture:** ConvNeXt-Tiny backbone
- **Optimization:** Automated search for Learning Rate, Batch Size, and Weight Decay via Optuna.
- **Distributed Training:** Configured with `accelerate` for high-performance training on RTX 5080.
- **Experiment Tracking:**
  - **TensorBoard:** Real-time monitoring of Train Loss, Val MAE, and Weight Decay impact.
  - **Optuna Dashboard:** Hyperparameter importance and study visualization.
- **Environment:** Managed with `uv` for ultra-fast dependency management and reproducible builds.

---

## Project Structure

```text
├── data/                 # Dataset and metadata (CSV)
├── data_analysis/        # Notebook about analysing data from UTKFace
├── logs/                 # TensorBoard event files
├── models/               # Optimized model checkpoints (.pth) with param-tracking names
├── optuna/               # SQLite database for optimization history
├── reports/              # Exported metrics plots
├── src/
│ ├── data_loader.py      # Custom Dataset with augmentation and strict split management
│ ├── model.py            # ConvNeXt-Tiny architecture implementation
│ ├── train_optuna.py     # Hyperparameter tuning script
│ ├── train.py            # Final training script with best parameters
│ ├── test.py             # Evaluation script for final Test Set (MAE & RMSE)
│ └── inference.py        # Single-image inference script for real-world testing
├── pyproject.toml        # Project dependencies
├── docker-compose.yml
├── dockerfile
└── README.md
```

---

## Installation & Setup

### Local Setup (Option 1)

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

### Docker Setup (Option 2 - Recommended)

To ensure a consistent environment with GPU support (make sure your docker is launched):

1. Build and start the container:

```bash
docker-compose up -d --build
```

2. Access the development environment:

```bash
docker exec -it age_dev bash
```

Note: All commands below (uv run...) should be executed inside the container.

---

## How to Run

### 1. Hyperparameter Optimization

Run the Optuna study to find the best settings:

```bash
uv run python src/train_optuna.py
```

Visualize the study:

```bash
uv run optuna-dashboard sqlite:///optuna/optuna_study.db --host 0.0.0.0
```

### 2. Final Training

Train the model with your selected parameters:

```bash
uv run python src/train.py
```

### 3. Monitoring Progress

Launch TensorBoard:

```bash
uv run tensorboard --logdir=logs --host=0.0.0.0 --port=6006
```

### 4. Evaluation & Inference

Run evaluation on the Test Set:

```bash
uv run python src/test.py
```

Predict age for a custom image:

```bash
uv run python src/inference.py --image /app/data/test/face_image.jpg
```

---

## Performance Metrics

- **Mean Absolute Error (MAE):** ~4.71 years on Test Set
- **Root Mean Squared Error (RMSE):** ~7.10 years
