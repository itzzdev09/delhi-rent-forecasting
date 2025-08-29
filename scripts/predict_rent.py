# scripts/predict_rent.py
import os
import sys
import pandas as pd
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import MODEL_DIR
from utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(MODEL_DIR, "rent_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

def load_model_and_preprocessor():
    """Load trained model and preprocessor pipeline."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}")

    log.info(f"Loading model from {MODEL_PATH} and preprocessor from {PREPROCESSOR_PATH}...")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

def preprocess_input(preprocessor, input_df):
    """Apply the saved preprocessor pipeline to new input data."""
    log.info("Preprocessing new input data...")
    X_processed = preprocessor.transform(input_df)
    
    # Feature names from one-hot encoder + numerical features
    cat_features = preprocessor.named_transformers_["categorical"]["onehot"].get_feature_names_out(
        preprocessor.transformers_[0][2]
    )
    numerical_features = preprocessor.transformers_[1][2]
    feature_names = list(cat_features) + list(numerical_features)

    return pd.DataFrame(X_processed.toarray(), columns=feature_names)

def predict_rent(model, X_processed):
    log.info(f"Predicting rent for {len(X_processed)} rows...")
    predictions = model.predict(X_processed)
    return pd.Series(predictions, name="predicted_rent")

def main():
    # Example new input CSV
    sample_input_csv = os.path.join(os.path.dirname(__file__), "sample_input.csv")
    if not os.path.exists(sample_input_csv):
        log.warning(f"{sample_input_csv} not found. Creating a dummy input CSV.")
        dummy_data = pd.DataFrame({
            "location": ["Dwarka", "Saket"],
            "bhk": [2, 3],
            "bathroom": [2, 3],
            "size": [1000, 1500],
            "furnishing": ["Furnished", "Semi-Furnished"],
            "tenant_preferred": ["Family", "Bachelor"],
            "point_of_contact": ["Alice", "Bob"]
        })
        dummy_data.to_csv(sample_input_csv, index=False)

    input_df = pd.read_csv(sample_input_csv)

    model, preprocessor = load_model_and_preprocessor()
    X_processed = preprocess_input(preprocessor, input_df)
    predictions = predict_rent(model, X_processed)
    
    input_df["predicted_rent"] = predictions
    print(input_df)

if __name__ == "__main__":
    main()
