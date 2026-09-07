import os
import torch
import torch.nn as nn
import torchvision.models as models

from utils import CLASSES  # single source of truth for the class list/order


def get_model_path():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Primary location: training/models/
    primary = os.path.join(base_dir, "training", "models", "gastric_resnet50.pth")
    if os.path.exists(primary):
        return primary
    # Legacy fallback: cancer_project/models/
    return os.path.join(base_dir, "cancer_project", "models", "gastric_resnet50.pth")


def get_fusion_model_path():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Primary location: training/models/
    primary = os.path.join(base_dir, "training", "models", "gastric_fusion.pth")
    if os.path.exists(primary):
        return primary
    # Legacy fallback: cancer_project/models/
    return os.path.join(base_dir, "cancer_project", "models", "gastric_fusion.pth")


def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model_path = get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\n[model_loader] gastric_resnet50.pth not found at:\n  {model_path}\n"
            "Place the model file in:  training/models/gastric_resnet50.pth"
        )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def load_feature_extractor():
    model = load_model()
    modules = list(model.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    feature_extractor.eval()
    return feature_extractor


class ClinicalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

    def forward(self, x):
        return self.net(x)


class GenomicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clinical_net = ClinicalMLP()
        self.genomic_net = GenomicMLP()
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 8 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(CLASSES))
        )

    def forward(self, img_features, clinical_data, genomic_data):
        clinical_features = self.clinical_net(clinical_data)
        genomic_features = self.genomic_net(genomic_data)
        combined = torch.cat([img_features, clinical_features, genomic_features], dim=1)
        return self.classifier(combined)


def load_fusion_model():
    fusion = FusionModel()
    fusion_path = get_fusion_model_path()
    if os.path.exists(fusion_path):
        fusion.load_state_dict(torch.load(fusion_path, map_location=torch.device("cpu")))
    fusion.eval()
    return fusionimport os
import torch
import torch.nn as nn
import torchvision.models as models

from utils import CLASSES  # single source of truth for the class list/order


def get_model_path():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Primary location: training/models/
    primary = os.path.join(base_dir, "training", "models", "gastric_resnet50.pth")
    if os.path.exists(primary):
        return primary
    # Legacy fallback: cancer_project/models/
    return os.path.join(base_dir, "cancer_project", "models", "gastric_resnet50.pth")


def get_fusion_model_path():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Primary location: training/models/
    primary = os.path.join(base_dir, "training", "models", "gastric_fusion.pth")
    if os.path.exists(primary):
        return primary
    # Legacy fallback: cancer_project/models/
    return os.path.join(base_dir, "cancer_project", "models", "gastric_fusion.pth")


def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model_path = get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\n[model_loader] gastric_resnet50.pth not found at:\n  {model_path}\n"
            "Place the model file in:  training/models/gastric_resnet50.pth"
        )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def load_feature_extractor():
    model = load_model()
    modules = list(model.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    feature_extractor.eval()
    return feature_extractor


class ClinicalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

    def forward(self, x):
        return self.net(x)


class GenomicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clinical_net = ClinicalMLP()
        self.genomic_net = GenomicMLP()
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 8 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(CLASSES))
        )

    def forward(self, img_features, clinical_data, genomic_data):
        clinical_features = self.clinical_net(clinical_data)
        genomic_features = self.genomic_net(genomic_data)
        combined = torch.cat([img_features, clinical_features, genomic_features], dim=1)
        return self.classifier(combined)


def load_fusion_model():
    fusion = FusionModel()
    fusion_path = get_fusion_model_path()
    if os.path.exists(fusion_path):
        fusion.load_state_dict(torch.load(fusion_path, map_location=torch.device("cpu")))
    fusion.eval()
    return fusion