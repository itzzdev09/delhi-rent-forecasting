
import joblib
import pandas as pd
from utils.config import MODEL_DIR
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


model_path = os.path.join(MODEL_DIR, "rent_model.pkl")
preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")


# Load model and preprocessor once
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

def predict_rent(input_data: dict) -> float:
    """
    Preprocess input dict and predict rent
    """
    df = pd.DataFrame([input_data])
    X_processed = preprocessor.transform(df)
    prediction = model.predict(X_processed)
    return round(float(prediction[0]), 2)
