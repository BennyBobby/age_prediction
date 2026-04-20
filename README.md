# Age Prediction with Deep Learning

A facial age estimation system using PyTorch and ConvNeXt-Tiny. Upload a photo and the app detects the face, predicts the age, and returns a confidence interval.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green) ![React](https://img.shields.io/badge/React-19-61dafb) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## How it works

```
Photo
  ↓
OpenCV Haar Cascade → detects & crops the face
  ↓
ConvNeXt-Tiny → 10 forward passes with MC Dropout
  ↓
Mean age + 95% confidence interval
```

**Best model performance:** MAE = 4.49 years · RMSE = 7.10 years on UTKFace test set

---

## Features

- **Face detection** : Haar cascade crops the face before inference and rejects images without a face
- **Confidence interval** : Monte Carlo Dropout over 10 passes returns a 95% range (e.g. _32 years, range 26–38_)
- **Face crop preview** : the detected face is returned as base64 and displayed in the UI
- **Hyperparameter optimization** : Optuna tunes learning rate, batch size, and weight decay
- **Distributed training** : Hugging Face Accelerate for multi-GPU support
- **Experiment tracking** : TensorBoard for loss/MAE curves, Optuna Dashboard for hyperparameters importance
- **REST API** : FastAPI with input validation, size limit (10 MB), structured logging
- **React frontend** : live preview, error messages, loading state
- **Test suite** : 17 pytest tests covering API, model architecture, and data pipeline
- **CI** : GitHub Actions runs tests on every push and pull request

---

## Project structure

```
├── api/
│   └── main.py               # FastAPI app (face detection, inference, confidence)
├── src/
│   ├── model.py              # ConvNeXt-Tiny architecture
│   ├── data_loader.py        # UTKFaceDataset with augmentation and 80/10/10 split
│   ├── train.py              # Training script (Accelerate + TensorBoard)
│   ├── train_optuna.py       # Hyperparameter search (Optuna, 15 trials)
│   ├── test.py               # Evaluation script (MAE & RMSE on test set)
│   └── inference.py          # CLI single-image inference
├── frontend/
│   ├── src/App.jsx           # React UI
│   └── dockerfile
├── tests/
│   ├── conftest.py           # Test fixtures and mocks
│   ├── test_api.py           # API tests (9 tests)
│   ├── test_model.py         # Model architecture tests (4 tests)
│   └── test_data_loader.py   # Data pipeline tests (5 tests)
├── data/                     # UTKFace dataset and metadata CSV
├── models/                   # Trained checkpoints (.pth)
├── logs/                     # TensorBoard event files
├── optuna/                   # SQLite study database
├── reports/                  # Training curve plots
├── docker-compose.yml
├── dockerfile
└── pyproject.toml
```

---

## Quick start

### Option 1 — Docker (recommended)

```bash
docker compose up -d --build
```

| Service            | URL                        |
| ------------------ | -------------------------- |
| Frontend           | http://localhost:5173      |
| API                | http://localhost:8000      |
| API docs (Swagger) | http://localhost:8000/docs |
| TensorBoard        | http://localhost:6006      |
| Optuna Dashboard   | http://localhost:8080      |

Start the API inside the container:

```bash
docker exec -it age_api_dev bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2 — Local

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Configure hardware (optional, for distributed training)
uv run accelerate config

# Start API
uv run uvicorn api.main:app --reload

# Start frontend
cd frontend && npm install && npm run dev
```

---

## Environment variables

| Variable       | Default                                                       | Description                             |
| -------------- | ------------------------------------------------------------- | --------------------------------------- |
| `MODEL_PATH`   | `models/convnext_tiny_bs64_lr2.90e-04_wd1.02e-04_mae4.49.pth` | Path to the model checkpoint            |
| `CORS_ORIGINS` | `http://localhost:5173`                                       | Comma-separated list of allowed origins |
| `VITE_API_URL` | `http://localhost:8000`                                       | API URL used by the frontend            |

---

## Training pipeline

### 1. Hyperparameter optimization

```bash
uv run python src/train_optuna.py
```

Runs 15 Optuna trials, tuning learning rate, batch size, and weight decay. Results saved to `optuna/optuna_study.db`.

```bash
# Visualize the study
uv run optuna-dashboard sqlite:///optuna/optuna_study.db --host 0.0.0.0
```

### 2. Training

```bash
uv run python src/train.py
```

Trains for 30 epochs with AdamW and L1 loss. Best checkpoint saved to `models/`. TensorBoard logs written to `logs/`.

```bash
uv run tensorboard --logdir=logs --host=0.0.0.0 --port=6006
```

### 3. Evaluation

```bash
uv run python src/test.py
```

### 4. Single-image inference (CLI)

```bash
uv run python src/inference.py --image /path/to/face.jpg
```

---

## API reference

### `POST /predict`

Accepts a face image, returns predicted age and confidence interval.

**Request:** `multipart/form-data` with a `file` field (image/\*, max 10 MB)

**Response:**

```json
{
  "filename": "photo.jpg",
  "predicted_age": 32.0,
  "confidence_interval": { "low": 26.3, "high": 37.7 },
  "face_crop": "data:image/jpeg;base64,...",
  "unit": "years",
  "device_used": "cuda"
}
```

**Error codes:**

| Code | Reason                        |
| ---- | ----------------------------- |
| 400  | File is not an image          |
| 413  | File exceeds 10 MB            |
| 422  | No face detected in the image |
| 500  | Internal inference error      |

### `GET /health`

```json
{ "status": "ok", "device": "cuda", "model_path": "models/..." }
```

---

## Tests

```bash
uv run pytest tests/ -v
```

The test suite uses a lightweight model with random weights (no checkpoint required) and mocks the face detector so tests run without real face photos.

---

## License

MIT — see [LICENSE](LICENSE)
