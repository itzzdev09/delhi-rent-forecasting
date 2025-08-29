import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from utils.config import *

logger = get_logger(__name__)

def load_data():
    logger.info(f"Loading processed data from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def main():
    X_train, X_test, y_train, y_test = load_data()

    logger.info("Training Linear Regression model...")
    model = train_model(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "rent_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
