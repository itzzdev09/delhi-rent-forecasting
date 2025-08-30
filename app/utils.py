import joblib
import pandas as pd
import os
from utils.config import MODEL_DIR

# Correct path — MODEL_DIR already includes saved_models
model_path = os.path.join(MODEL_DIR, "rent_model.pkl")
preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

# Load model & preprocessor
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

def predict_rent(input_data: dict) -> float:
    df = pd.DataFrame([input_data])
    X_processed = preprocessor.transform(df)
    prediction = model.predict(X_processed)
    return round(float(prediction[0]), 2)

def get_dropdown_options():
    from utils.config import PROCESSED_DATA_PATH
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return {
        "locations": sorted(df['location'].dropna().unique().tolist()),
        "house_types": sorted(df['house_type'].dropna().unique().tolist()),
        "house_sizes": sorted(df['house_size'].dropna().unique().tolist()),
        "furnishing": sorted(df['furnishing'].dropna().unique().tolist()) if 'furnishing' in df.columns else [],
        "tenant_preferred": sorted(df['tenant_preferred'].dropna().unique().tolist()) if 'tenant_preferred' in df.columns else []
    }
