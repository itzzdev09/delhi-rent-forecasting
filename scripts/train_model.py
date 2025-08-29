# scripts/train_model.py
"""
Train ML model for Delhi Rent Prediction:
- Loads processed dataset
- Splits into train/test
- Trains Linear Regression
- Saves trained model and preprocessor
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from utils.config import *

logger = get_logger(__name__)

def load_data():
    """
    Load processed data and split into train/test sets
    """
    logger.info(f"Loading processed data from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test

def build_preprocessor(X):
    """
    Build preprocessing pipeline (OneHot + Scaling)
    """
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_transformer, categorical_features),
        ("numerical", numerical_transformer, numerical_features)
    ])
    return preprocessor

def train_model(X_train, y_train):
    """
    Train Linear Regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Build preprocessor and transform
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info("Training Linear Regression model...")
    model = train_model(X_train_processed, y_train)

    # Save model and preprocessor
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "saved_models", "rent_model.pkl")
    preprocessor_path = os.path.join(MODELS_DIR, "saved_models", "preprocessor.pkl")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Preprocessor saved to {preprocessor_path}")

if __name__ == "__main__":
    main()
