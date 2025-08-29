# scripts/train_model.py
"""
Train ML models for Delhi House Rent Prediction.
- Loads processed dataset
- Splits into train/test
- Trains baseline models (Linear Regression, Random Forest)
- Evaluates models and selects best one
- Saves trained model
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

from utils.config import Config
logger.info(f"Loading processed data from {Config.PROCESSED_DATA_PATH}...")



def load_data():
    logger.info(f"Loading processed data from {Config.PROCESSED_DATA_PATH}...")
    df = pd.read_csv(Config.PROCESSED_DATA_PATH)

    X = df.drop(Config.TARGET_COLUMN, axis=1)
    y = df[Config.TARGET_COLUMN]

    return train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)


def train_model(X_train, y_train):
    """
    Train a simple Linear Regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def main():
    X_train, X_test, y_train, y_test = load_data()

    logger.info("Training model...")
    model = train_model(X_train, y_train)

    # Save model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_path = os.path.join(config.MODELS_DIR, "rent_model.pkl")
    joblib.dump(model, model_path)

    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
