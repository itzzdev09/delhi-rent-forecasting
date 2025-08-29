# scripts/train_model.py
"""
Train ML models for Delhi House Rent Prediction.
- Loads processed dataset
- Splits into train/test
- Trains baseline models (Linear Regression, Random Forest)
- Evaluates models and selects best one
- Saves trained model
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from utils import config, logger

log = logger.get_logger(__name__)


def load_data(file_path=config.PROCESSED_DATA_PATH):
    """Load processed dataset."""
    log.info(f"Loading processed data from {file_path}...")
    return pd.read_csv(file_path)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train models and evaluate performance."""
    results = {}
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    for name, model in models.items():
        log.info(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        results[name] = {"model": model, "r2": r2, "rmse": rmse, "mae": mae}
        log.info(f"{name} - R2: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    return results


def save_best_model(results):
    """Save the best model based on R² score."""
    best_model_name = max(results, key=lambda k: results[k]["r2"])
    best_model = results[best_model_name]["model"]

    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, config.MODEL_PATH)

    log.info(f"Best model '{best_model_name}' saved to {config.MODEL_PATH}")


def main():
    df = load_data()

    # Features & Target
    X = df.drop("Rent", axis=1)
    y = df["Rent"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    save_best_model(results)


if __name__ == "__main__":
    main()
