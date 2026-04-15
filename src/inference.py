import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import get_model
import argparse
import os


def predict(image_path, model_path):
    img_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found.")
        return

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    model = get_model()
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.to(device)
    model.eval()

    print(f"Running inference on {device}...")
    with torch.no_grad():
        output = model(input_tensor)
        predicted_age = output.item()
    print(f"\nFILE: {os.path.basename(image_path)}")
    print(f"PREDICTED AGE: {predicted_age:.2f} years")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict age from a face image")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/convnext_tiny_bs64_lr2.90e-04_wd1.02e-04_mae4.49.pth",
        help="Path to the .pth model file",
    )

    args = parser.parse_args()
    predict(args.image, args.model)
