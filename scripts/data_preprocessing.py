import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from faker import Faker
import random
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import get_logger
from utils.config import *

log = get_logger(__name__)

def preprocess_data():
    # Load raw dataset
    log.info(f"Loading raw dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Forward-fill missing values
    log.info("Cleaning data...")
    df = df.ffill()

    # Synthetic data
    log.info(f"Generating {SYNTHETIC_DATA_ROWS} synthetic rows with Faker...")
    faker = Faker()
    synthetic_data = []
    for _ in range(SYNTHETIC_DATA_ROWS):
        synthetic_data.append({
            "location": faker.city(),
            "bhk": random.randint(1, 5),
            "bathroom": random.randint(1, 4),
            "size": random.randint(500, 2500),
            "rent": random.randint(5000, 50000),
            "furnishing": random.choice(["Furnished", "Semi-Furnished", "Unfurnished"]),
            "tenant_preferred": random.choice(["Family", "Bachelor", "Company"]),
            "point_of_contact": faker.name()
        })
    df = pd.concat([df, pd.DataFrame(synthetic_data)], ignore_index=True)
    log.info(f"New dataset size after augmentation: {df.shape}")

    # Fill NaNs in target column with median
    if df[TARGET_COLUMN].isna().any():
        log.warning(f"NaN values detected in target column '{TARGET_COLUMN}'. Filling with median...")
        df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(df[TARGET_COLUMN].median())

    # Fill NaNs in features
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        log.warning(f"NaN values detected in columns: {nan_cols}. Filling with forward fill...")
        df = df.ffill()

    # Drop constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        log.warning(f"Dropping constant columns (no variance): {constant_cols}")
        df = df.drop(columns=constant_cols)

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Identify categorical and numerical
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    log.info(f"Encoding categorical features: {categorical_features}")
    log.info(f"Scaling numerical features: {numerical_features}")

    # Pipeline
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

    X_processed = preprocessor.fit_transform(X)

    # Convert to DataFrame
    try:
        # if sparse matrix
        X_processed = X_processed.toarray()
    except AttributeError:
        pass  # already dense

    cat_features = preprocessor.named_transformers_["categorical"]["onehot"].get_feature_names_out(categorical_features)
    feature_names = list(cat_features) + numerical_features
    X_processed = pd.DataFrame(X_processed, columns=feature_names)

    # Merge target back
    processed_df = pd.concat([X_processed, y.reset_index(drop=True)], axis=1)

    # Save
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    log.info(f"Processed dataset saved at {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    preprocess_data()
