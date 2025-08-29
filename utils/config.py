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

# -------------------------------
# Training Parameters
# -------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

# Target column for regression
TARGET_COLUMN = "rent"

# Synthetic data augmentation
SYNTHETIC_DATA_ROWS = 1000

# LSTM & Regression placeholders
LSTM_PARAMS = {
    "input_size": 10,      
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": False,
}

REGRESSION_PARAMS = {
    "fit_intercept": True,
    "normalize": False
}
