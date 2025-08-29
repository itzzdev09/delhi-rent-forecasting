import os

import os

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "delhi_rentals.csv")
    PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "processed_rentals.csv")


    MODELS_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Target variable(s)
    TARGET_RENT = "rent"
    TARGET_PRICE = "price"


    # -------------------------------
    # Training Parameters
    # -------------------------------
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50

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

    # -------------------------------
    # Synthetic Data Parameters
    # -------------------------------
    SYNTHETIC_DATA_ROWS = 1000   # number of fake rows to generate
