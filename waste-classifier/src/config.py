"""Central configuration: paths and hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "waste_model.keras"
TFLITE_PATH = MODELS_DIR / "waste_model.tflite"
LABELS_PATH = MODELS_DIR / "labels.json"

DISPOSAL_GUIDE_PATH = ROOT / "configs" / "disposal_guide.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5
FINE_TUNE_AT = 100  # unfreeze MobileNetV2 layers from this index onward
