# scripts/predict_rent.py
"""
Predict rent for new house listings.
- Loads trained model and preprocessor
- Accepts input CSV or DataFrame
- Handles missing columns gracefully
- Outputs predictions
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.config import *

logger = get_logger(__name__)


def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "rent_model.pkl")
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")

    logger.info(f"Loading model from {model_path} and preprocessor from {preprocessor_path}...")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def preprocess_input(preprocessor, df):
    """
    Ensure all columns required by the preprocessor exist.
    Missing numeric columns -> fill 0
    Missing categorical columns -> fill 'Unknown'
    """
    # Expected columns
    all_columns = []
    for name, transformer, cols in preprocessor.transformers_:
        all_columns.extend(cols)
    
    missing_cols = set(all_columns) - set(df.columns)
    for col in missing_cols:
        if col in df.select_dtypes(include=["object"]).columns:
            df[col] = "Unknown"
        else:
            df[col] = 0

    df = df[all_columns]  # reorder columns
    X_processed = preprocessor.transform(df)
    return X_processed


def predict(df):
    model, preprocessor = load_artifacts()
    X_processed = preprocess_input(preprocessor, df)
    predictions = model.predict(X_processed)
    df["predicted_rent"] = predictions
    return df


def main():
    # Example: load input CSV
    input_path = os.path.join(os.path.dirname(__file__), "sample_input.csv")
    
    if not os.path.exists(input_path):
        logger.warning(f"{input_path} not found. Creating a dummy input CSV.")
        dummy = pd.DataFrame({
            "location": ["Delhi"], "bhk": [2], "bathroom": [2], "size": [1200],
            "furnishing": ["Semi-Furnished"], "tenant_preferred": ["Family"],
            "point_of_contact": ["John Doe"]
        })
        dummy.to_csv(input_path, index=False)
    
    input_df = pd.read_csv(input_path)
    logger.info("Preprocessing new input data...")
    output_df = predict(input_df)
    
    logger.info("Predictions:")
    print(output_df)


if __name__ == "__main__":
    main()
