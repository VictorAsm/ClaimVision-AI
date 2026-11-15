import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Raíz dinámica del proyecto (un nivel arriba de backend/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Carpetas del módulo de ML
ML_DIR = BASE_DIR / "ml"
DATA_DIR = ML_DIR / "data"
WEIGHTS_PATH = ML_DIR / "weights"

# Rutas de datos procesados y crudos
PROCESSED_PATH_ORIGINAL = DATA_DIR / "processed" / "original"
PROCESSED_PATH_RETRAIN = DATA_DIR / "processed" / "retrain"
RAW_PATH_ORIGINAL = DATA_DIR / "raw" / "original"
RAW_PATH_RETRAIN = DATA_DIR / "raw" / "retrain"
NO_CLASSIFY_PATH = DATA_DIR / "no_classify"

# Modelos y Checkpoints (Clasificación)
CLASSIFICATION_MODEL_PATH = WEIGHTS_PATH / "classification" / "neural_network_model.keras"
CLASSIFICATION_CHECKPOINT_PATH = str(WEIGHTS_PATH / "classification" / "checkpoints" / "save_at_{epoch}.keras")
CLASSIFICATION_LABEL_PATH = PROCESSED_PATH_ORIGINAL / "csv" / "labels-damaged.csv"

# Modelos y Checkpoints (Detección de Objetos)
OBJECT_DETECTION_MODEL_PATH = WEIGHTS_PATH / "object_detection" / "object_detection_model.pth"
OBJECT_DETECTION_CHECKPOINT_PATH = WEIGHTS_PATH / "object_detection" / "checkpoints"
OBJECT_DETECTION_LABEL_PATH = PROCESSED_PATH_ORIGINAL / "json" / "labels-damaged.json"

# Credenciales de API de Google Gemini (leídas desde el entorno / archivo .env)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
