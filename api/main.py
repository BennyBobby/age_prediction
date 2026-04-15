import torch
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from src.model import get_model
from torchvision import transforms

app = FastAPI(
    title="Age Prediction API",
    description="API to predict age from a face image using ConvNeXt-Tiny",
    version="1.0.0",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/convnext_tiny_bs64_lr2.90e-04_wd1.02e-04_mae4.49.pth"

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = get_model()
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"Model successfully loaded on {DEVICE}")
except Exception as e:
    print(f"Error loading the model: {e}")


@app.get("/")
def read_root():
    return {"message": "Age Prediction API is running. Go to /docs for Swagger UI."}


@app.post("/predict")
async def predict_age(file: UploadFile = File(...)):
    """Predict age from an uploaded image file."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_age = output.item()

        return {
            "filename": file.filename,
            "predicted_age": round(predicted_age, 2),
            "unit": "years",
            "device_used": str(DEVICE),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
