"""Single source of truth for paths and constants.

Nothing here is dataset-specific: the image class list is discovered from the
folder names of whatever image dataset you point the scripts at, and is stored
inside the trained checkpoint so training and serving can never disagree.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

SEED = 42
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Folder names treated as "tumour tissue" for the tumour-probability readout.
TUMOUR_CLASSES = ("TUM",)

# Predictions whose top probability is below this are flagged as low-confidence.
LOW_CONFIDENCE = 0.60

# Minimum fraction of the model's genes that an uploaded expression file must
# contain before the genomic modality is considered "present".
MIN_GENOMIC_COVERAGE = 0.50

GENE_PREFIX = "g__"
CLINICAL_FEATURES = [
    "age_z", "age_missing", "sex_female", "sex_missing", "stage_z", "stage_missing",
]
CLINICAL_DIM = len(CLINICAL_FEATURES)

DISCLAIMER = (
    "Research prototype. Not clinically validated and not for diagnosis or "
    "treatment decisions. All outputs must be reviewed by a qualified pathologist."
)

# Plain-language names for common tissue folder codes (unknown codes are shown as-is).
TISSUE = {
    "ADI": "adipose (fat) tissue", "BACK": "background / no tissue", "DEB": "debris / cell fragments",
    "LYM": "lymphocytes (immune cells)", "MUC": "mucus", "MUS": "smooth muscle",
    "NOR": "normal mucosa", "NORM": "normal mucosa", "STR": "stroma / connective tissue",
    "TUM": "tumour tissue",
}
