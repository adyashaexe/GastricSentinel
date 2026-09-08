from torchvision import transforms
from PIL import Image

# Alphabetical order, matching ImageFolder's class-to-index assignment: Abnormal=0, Normal=1
CLASSES = ['Abnormal', 'Normal']

def get_transform():
    """
    Returns the exact same transform used during validation.
    Essential for accurate predictions.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def preprocess_image(image_path):
    """
    Loads an image and prepares it for the model.
    """
    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    return transform(image)

def generate_report(class_index, confidence):
    """
    2-Tier 'Traffic Light' Logic for binary Abnormal/Normal classification.
    """
    detected_class = CLASSES[class_index]
    conf_percent = confidence * 100

    if detected_class == 'Abnormal':
        return {
            "tier": "SUSPICIOUS",
            "color": "🔴 RED",
            "diagnosis": "Abnormal Gastric Tissue",
            "details": f"Model is {conf_percent:.1f}% confident this tissue is abnormal.",
            "recommendation": "Pathological review recommended."
        }

    # detected_class == 'Normal'
    return {
        "tier": "NEGATIVE",
        "color": "🟢 GREEN",
        "diagnosis": "Normal Gastric Tissue",
        "details": f"Confidence: {conf_percent:.1f}%",
        "recommendation": "No abnormalities detected in this view."
    }