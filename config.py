import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'crop_disease_model.keras')
CLASS_LABELS_PATH = os.path.join(MODEL_DIR, 'class_labels.json')

# Dataset
DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'PlantVillage')

# Static
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
GRAPHS_FOLDER = os.path.join(STATIC_DIR, 'graphs')

# Database
DB_DIR = os.path.join(BASE_DIR, 'database')
DISEASE_INFO_PATH = os.path.join(DB_DIR, 'disease_information.json')
HISTORY_DB_PATH = os.path.join(DB_DIR, 'history.db')

# Reports
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Upload settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Model input
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
FINE_TUNE_EPOCHS = 10
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

# Flask
SECRET_KEY = 'cropcare-ai-secret-2024'
DEBUG = True

# Training split
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
