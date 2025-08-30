import joblib
import pandas as pd
from utils.config import RENT_MODEL_PATH, PREPROCESSOR_PATH, TARGET_COLUMN
from utils.logger import get_logger

log = get_logger(__name__)

# Load model & preprocessor once
log.info("Loading model and preprocessor...")
model = joblib.load(RENT_MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
log.info("Model and preprocessor loaded successfully.")

def predict_rent(input_df: pd.DataFrame) -> pd.Series:
    """
    input_df: dataframe with columns matching training features
    returns: predicted rent
    """
    # Preprocess
    X_processed = preprocessor.transform(input_df)
    
    # Model prediction
    predictions = model.predict(X_processed)
    
    return pd.Series(predictions)
