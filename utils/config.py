import os

# -------------------------------
# Project Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "delhi_rentals.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "processed_rentals.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -------------------------------
# Training Parameters
# -------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

# -------------------------------
# Synthetic Data Parameters
# -------------------------------
SYNTHETIC_DATA_ROWS = 1000

# -------------------------------
# Columns
# -------------------------------
TARGET_COLUMN = "price"
FEATURE_COLUMNS = [
    'house_type', 'house_size', 'location', 'city', 'latitude', 'longitude',
    'numBathrooms', 'numBalconies', 'isNegotiable', 'priceSqFt', 'verificationDate',
    'description', 'SecurityDeposit', 'Status'
]
