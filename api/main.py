import os
import io
import base64
import logging
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from src.model import get_model
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Age Prediction API",
    description="API to predict age from a face image using ConvNeXt-Tiny",
    version="1.0.0",
)

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "models/convnext_tiny_bs64_lr2.90e-04_wd1.02e-04_mae4.49.pth")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
N_MC_PASSES = 10

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model = get_model()
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
logger.info("Model loaded on %s from %s", DEVICE, MODEL_PATH)


def _detect_and_crop_face(img: Image.Image) -> Image.Image:
    """Detect the largest face and return a cropped image with 20% margin."""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        raise HTTPException(status_code=422, detail="No face detected in the image.")

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face by area
    img_w, img_h = img.size
    margin_x = w * 0.2
    margin_y = h * 0.2
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)
    return img.crop((x1, y1, x2, y2))


def _predict_with_uncertainty(tensor: torch.Tensor) -> tuple[float, float]:
    """Run N_MC_PASSES forward passes with stochastic depth active, return (mean, std)."""
    model.train()  # activates stochastic depth for uncertainty estimation
    preds = []
    with torch.no_grad():
        for _ in range(N_MC_PASSES):
            preds.append(model(tensor).item())
    model.eval()
    return float(np.mean(preds)), float(np.std(preds))


@app.get("/")
def read_root():
    return {"message": "Age Prediction API is running. Go to /docs for Swagger UI."}


@app.get("/health")
def health_check():
    return {"status": "ok", "device": str(DEVICE), "model_path": MODEL_PATH}


@app.post("/predict")
async def predict_age(file: UploadFile = File(...)):
    """Predict age from an uploaded face image."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        face = _detect_and_crop_face(img)
        input_tensor = transform(face).unsqueeze(0).to(DEVICE)
        mean_age, std_age = _predict_with_uncertainty(input_tensor)

        buf = io.BytesIO()
        face.save(buf, format="JPEG")
        face_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        logger.info("Predicted age %.1f ± %.1f for %s", mean_age, std_age, file.filename)
        return {
            "filename": file.filename,
            "predicted_age": round(mean_age, 1),
            "confidence_interval": {
                "low": round(mean_age - 2 * std_age, 1),
                "high": round(mean_age + 2 * std_age, 1),
            },
            "face_crop": f"data:image/jpeg;base64,{face_b64}",
            "unit": "years",
            "device_used": str(DEVICE),
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Inference failed for file %s", file.filename)
        raise HTTPException(status_code=500, detail="Internal inference error.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
