from PIL import Image
import torch
from torchvision import transforms

CLASSES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


def preprocess_image(file):
    transform = get_transform()
    image = Image.open(file).convert("RGB")
    return transform(image).unsqueeze(0)


def clinical_to_tensor(age, gender, stage):
    gender_map = {"male": 0, "female": 1}
    stage_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4}

    try:
        age = float(age)
    except Exception:
        age = 0.0

    gender_val = gender_map.get(str(gender).lower(), 0)
    stage_val = stage_map.get(str(stage).lower(), 1)

    return torch.tensor([age, gender_val, stage_val]).float().unsqueeze(0)


def genomic_to_tensor(genes):
    MAX_GENES = 20
    cleaned = []
    for g in genes:
        try:
            cleaned.append(float(g))
        except Exception:
            cleaned.append(0.0)

    if len(cleaned) < MAX_GENES:
        cleaned += [0.0] * (MAX_GENES - len(cleaned))
    cleaned = cleaned[:MAX_GENES]

    tensor = torch.tensor(cleaned).float()
    tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-6)
    return tensor.unsqueeze(0)


def generate_report(class_index, confidence):
    detected_class = CLASSES[class_index]
    conf_percent = confidence * 100

    if detected_class == "TUM":
        return {
            "tier": "CRITICAL",
            "color": "RED",
            "diagnosis": "Gastric Adenocarcinoma (Cancer)",
            "details": f"Model is {conf_percent:.1f}% confident this is tumor tissue.",
            "recommendation": "Immediate pathological review required. Urgent oncology referral."
        }

    if detected_class == "STR":
        return {
            "tier": "SUSPICIOUS",
            "color": "YELLOW",
            "diagnosis": "Cancer-Associated Stroma",
            "details": f"Model detected abnormal connective tissue ({conf_percent:.1f}%).",
            "recommendation": "High-risk area. Check adjacent tissue for tumor cells."
        }

    friendly_names = {
        "ADI": "Adipose (Fat Tissue)",
        "DEB": "Debris / Cellular Fragments",
        "LYM": "Lymphocytes (Immune Cells)",
        "MUC": "Mucosa (Stomach Lining)",
        "MUS": "Smooth Muscle",
        "NORM": "Normal Mucosa"
    }

    if detected_class == "BACK":
        return {
            "tier": "INVALID",
            "color": "GRAY",
            "diagnosis": "Background / No Tissue Detected",
            "details": f"Model is {conf_percent:.1f}% confident this region contains no tissue.",
            "recommendation": "No tissue present in this region. Re-scan or select a different area of the slide."
        }

    if detected_class in friendly_names:
        return {
            "tier": "NEGATIVE",
            "color": "GREEN",
            "diagnosis": f"Healthy Tissue ({friendly_names[detected_class]})",
            "details": f"Model confidence: {conf_percent:.1f}%",
            "recommendation": "No malignancies detected in this image."
        }

    return {
        "tier": "INVALID",
        "color": "GRAY",
        "diagnosis": "Non-Tissue / Artifact",
        "details": f"Detected class: {detected_class}",
        "recommendation": "Image rejected. Upload a clearer histopathology image."
    }