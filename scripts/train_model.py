# scripts/train_model.py
"""
Train ML models for Delhi House Rent Prediction.
- Loads processed dataset
- Splits into train/test
- Trains Linear Regression model
- Saves model in models/saved_models
"""

import sys, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logger import get_logger
from utils.config import *

log = get_logger(__name__)


def load_data():
    log.info(f"Loading processed data from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Drop rows with NaN target just in case
    df = df.dropna(subset=[TARGET_COLUMN])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)


def train_model(X_train, y_train):
    log.info("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def main():
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "saved_models", "rent_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    log.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
