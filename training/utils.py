from torchvision import transforms
from PIL import Image

# The exact order matters! It must match the alphabetical order of folders.
CLASSES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

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
    The 4-Tier 'Traffic Light' Logic.
    """
    detected_class = CLASSES[class_index]
    conf_percent = confidence * 100

    # --- TIER 1: DANGER (Red) ---
    if detected_class == 'TUM':
        return {
            "tier": "CRITICAL",
            "color": "🔴 RED",
            "diagnosis": "Gastric Adenocarcinoma (Cancer)",
            "details": f"Model is {conf_percent:.1f}% confident this is tumor tissue.",
            "recommendation": "Immediate pathological review required."
        }

    # --- TIER 2: WARNING (Yellow) ---
    elif detected_class == 'STR':
        return {
            "tier": "SUSPICIOUS",
            "color": "🟡 YELLOW",
            "diagnosis": "Cancer-Associated Stroma",
            "details": f"Model detected abnormal connective tissue ({conf_percent:.1f}%).",
            "recommendation": "High risk area. Check adjacent tissue for tumor cells."
        }

    # --- TIER 3: HEALTHY (Green) ---
    elif detected_class in ['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM']:
        friendly_names = {
            'ADI': 'Adipose (Fat)',
            'DEB': 'Debris / Cellular Fragments',
            'LYM': 'Lymphocytes (Immune Cells)',
            'MUC': 'Mucosa (Stomach Lining)',
            'MUS': 'Smooth Muscle',
            'NORM': 'Normal Mucosa'
        }
        return {
            "tier": "NEGATIVE",
            "color": "🟢 GREEN",
            "diagnosis": f"Healthy Tissue ({friendly_names[detected_class]})",
            "details": f"Confidence: {conf_percent:.1f}%",
            "recommendation": "No malignancies detected in this view."
        }

    # --- TIER 4: INVALID (Gray) ---
    elif detected_class == 'BACK':
        return {
            "tier": "INVALID",
            "color": "⚪ GRAY",
            "diagnosis": "Background / No Tissue Detected",
            "details": f"Model is {conf_percent:.1f}% confident this region contains no tissue.",
            "recommendation": "No tissue present. Re-scan or select a different area of the slide."
        }

    else:
        return {
            "tier": "INVALID",
            "color": "⚪ GRAY",
            "diagnosis": "Non-Tissue / Artifact",
            "details": f"Detected: {detected_class}",
            "recommendation": "Image rejected. Please upload a clear tissue scan."
        }