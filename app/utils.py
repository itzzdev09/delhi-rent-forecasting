import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import MODEL_DIR, DATA_DIR

saved_dir = os.path.join(MODEL_DIR, "saved_models")
model_path = os.path.join(saved_dir, "rent_model.pkl")
preprocessor_path = os.path.join(saved_dir, "preprocessor.pkl")

# Load model + preprocessor
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# Load processed dataset (assuming you saved it after cleaning)
data_path = os.path.join(DATA_DIR, "processed_data.csv")
df = pd.read_csv(data_path)

# Dynamic dropdowns from dataset
DROPDOWN_OPTIONS = {
    "house_type": sorted(df["house_type"].dropna().unique().tolist()),
    "house_size": sorted(df["house_size"].dropna().unique().tolist()),
    "location": sorted(df["location"].dropna().unique().tolist()),
    "furnishing": sorted(df["furnishing"].dropna().unique().tolist()),
    "tenant_preferred": sorted(df["tenant_preferred"].dropna().unique().tolist()),
}

# Optional fields with defaults
DEFAULTS = {
    "furnishing": DROPDOWN_OPTIONS["furnishing"][0],
    "tenant_preferred": DROPDOWN_OPTIONS["tenant_preferred"][0],
    "numBalconies": 0,
    "numBathrooms": 1,
    "latitude": df["latitude"].mean() if "latitude" in df else 0.0,
    "longitude": df["longitude"].mean() if "longitude" in df else 0.0,
}

def preprocess_input(input_data: dict) -> pd.DataFrame:
    # Fill missing optional fields
    for key, value in DEFAULTS.items():
        if key not in input_data or input_data[key] is None:
            input_data[key] = value

    # Validate against dropdowns
    for key, valid_list in DROPDOWN_OPTIONS.items():
        if key in input_data and input_data[key] not in valid_list:
            input_data[key] = valid_list[0]  # fallback to first option

    return pd.DataFrame([input_data])

def predict_rent(input_data: dict) -> float:
    df = preprocess_input(input_data)
    X_processed = preprocessor.transform(df)
    prediction = model.predict(X_processed)
    return round(float(prediction[0]), 2)

def get_dropdown_options():
    return DROPDOWN_OPTIONS
