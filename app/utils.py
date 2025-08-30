import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import MODEL_DIR

saved_dir = os.path.join(MODEL_DIR, "saved_models")
model_path = os.path.join(saved_dir, "rent_model.pkl")
preprocessor_path = os.path.join(saved_dir, "preprocessor.pkl")

# Load model and preprocessor once
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# Define dropdown options (can also load from processed dataset if dynamic)
DROPDOWN_OPTIONS = {
    "house_type": ["Apartment", "Independent House", "Studio"],
    "house_size": ["1 BHK", "2 BHK", "3 BHK", "4 BHK"],
    "location": ["Dwarka", "Rohini", "Janakpuri", "Karol Bagh"],
    "furnishing": ["Furnished", "Semi-Furnished", "Unfurnished"],
    "tenant_preferred": ["Family", "Bachelor", "Company"]
}

# Optional fields with defaults
DEFAULTS = {
    "furnishing": "Unfurnished",
    "tenant_preferred": "Family",
    "numBalconies": 0,
    "numBathrooms": 1,
    "latitude": 0.0,
    "longitude": 0.0
}

def preprocess_input(input_data: dict) -> pd.DataFrame:
    """
    Fill missing optional fields and return DataFrame ready for preprocessor
    """
    for key, value in DEFAULTS.items():
        if key not in input_data:
            input_data[key] = value

    # Optionally validate categorical dropdowns
    for key, valid_list in DROPDOWN_OPTIONS.items():
        if key in input_data and input_data[key] not in valid_list:
            input_data[key] = valid_list[0]  # fallback to first option

    df = pd.DataFrame([input_data])
    return df

def predict_rent(input_data: dict) -> float:
    df = preprocess_input(input_data)
    X_processed = preprocessor.transform(df)
    prediction = model.predict(X_processed)
    return round(float(prediction[0]), 2)

def get_dropdown_options():
    return DROPDOWN_OPTIONS
