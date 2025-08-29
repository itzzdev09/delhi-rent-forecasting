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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer   # 👈 added
import joblib

from utils import config, logger


def load_data():
    logger.info(f"Loading processed data from {config.PROCESSED_DATA_PATH}...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH)

    X = df.drop(columns=["rent"])
    y = df["rent"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    # ✅ Step 1: Handle missing values with imputation
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # ✅ Step 2: Train model
    logger.info("Training LinearRegression...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # ✅ Step 3: Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Evaluation Results - MSE: {mse:.2f}, R2: {r2:.2f}")
    return model, mse, r2, imputer


def main():
    X_train, X_test, y_train, y_test = load_data()
    model, mse, r2, imputer = train_and_evaluate(X_train, X_test, y_train, y_test)

    # ✅ Save both model and imputer (so we can preprocess future inputs consistently)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(imputer, config.IMPUTER_PATH)  # 👈 save imputer separately
    logger.info(f"Model saved to {config.MODEL_PATH}")
    logger.info(f"Imputer saved to {config.IMPUTER_PATH}")


if __name__ == "__main__":
    main()
