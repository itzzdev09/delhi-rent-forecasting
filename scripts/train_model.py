# scripts/train_model.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from utils.config import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

log = get_logger(__name__)

def load_data():
    log.info(f"Loading processed data from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop("rent", axis=1)
    y = df["rent"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def main():
    X_train, X_test, y_train, y_test = load_data()
    log.info("Training Linear Regression model...")
    model = train_model(X_train, y_train)

    # Save model
    model_dir = os.path.join(os.path.dirname(PROCESSED_DATA_PATH), "..", "models", "saved_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rent_model.pkl")
    joblib.dump(model, model_path)
    log.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
