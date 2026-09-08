import torch
import torch.nn as nn
from torchvision import models
import argparse
import os
import sys

from utils import CLASSES, preprocess_image, generate_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT, 'model', 'gastric_resnet50.pkl')


def load_trained_model(device):
    """
    Reconstructs the ResNet50 architecture and loads your trained weights.
    Handles either a full pickled nn.Module or a state_dict checkpoint,
    same as train_fusion_model.py, since gastric_resnet50.pkl may be either.
    """
    print(f"Loading model from {MODEL_PATH}...")

    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model file not found. Run training/train.py first!")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_trained_model(device)

    try:
        input_tensor = preprocess_image(image_path).to(device).unsqueeze(0)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        report = generate_report(predicted_idx.item(), confidence.item())

    print("\n" + "=" * 40)
    print(f"  DIAGNOSTIC REPORT")
    print("=" * 40)
    print(f"STATUS:  {report['color']} {report['tier']}")
    print(f"FINDING: {report['diagnosis']}")
    print(f"NOTE:    {report['details']}")
    print(f"ACTION:  {report['recommendation']}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gastric Tissue Classifier")
    parser.add_argument('--image', type=str, required=True, help="Path to the tissue image")
    args = parser.parse_args()

    predict(args.image)