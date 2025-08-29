# scripts/data_preprocessing.py
"""
Data preprocessing pipeline for Delhi House Rent Prediction.
- Cleans missing/duplicate values
- Encodes categorical features
- Scales numerical features
- Generates synthetic data using Faker
- Saves final processed dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from faker import Faker
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils import config, logger

log = logger.get_logger(__name__)


def load_raw_data(file_path=config.RAW_DATA_PATH):
    """Load raw dataset from CSV."""
    log.info(f"Loading raw dataset from {file_path}...")
    return pd.read_csv(file_path)


def clean_data(df):
    """Handle missing values and remove duplicates."""
    log.info("Cleaning data...")
    df = df.drop_duplicates()
    df = df.fillna(method="ffill")  # forward fill for simplicity
    return df


def encode_features(df):
    """Encode categorical columns (basic version)."""
    log.info("Encoding categorical features...")

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def scale_features(df):
    log.info("Scaling numerical features...")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "Rent" in num_cols:
        num_cols.remove("Rent")  # don’t scale target

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df



def augment_data(df, n_samples=500):
    """Use Faker to generate synthetic rental data for Delhi."""
    log.info(f"Generating {n_samples} synthetic rows with Faker...")
    fake = Faker()

    synthetic_data = []
    for _ in range(n_samples):
        row = {
            "Location": fake.random_element(
                elements=["Dwarka", "Saket", "Rohini", "Karol Bagh", "Hauz Khas"]
            ),
            "BHK": fake.random_int(min=1, max=5),
            "Size": fake.random_int(min=500, max=3000),
            "Rent": fake.random_int(min=5000, max=100000),
        }
        synthetic_data.append(row)

    df_synthetic = pd.DataFrame(synthetic_data)
    df = pd.concat([df, df_synthetic], ignore_index=True)

    log.info(f"New dataset size after augmentation: {df.shape}")
    return df


def preprocess_pipeline():
    """Run full preprocessing pipeline."""
    df = load_raw_data()
    df = clean_data(df)
    df = augment_data(df, n_samples=1000)
    df = encode_features(df)
    df = scale_features(df)

    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    log.info(f"Processed dataset saved at {config.PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    preprocess_pipeline()
