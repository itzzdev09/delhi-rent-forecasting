# streamlit_app.py
"""
Streamlit UI for Delhi Rent Prediction.

Assumptions:
- utils/config.py defines MODEL_DIR and PROCESSED_DATA_PATH
- models/saved_models contains rent_model.pkl and preprocessor.pkl
- processed dataset exists at PROCESSED_DATA_PATH and contains the original feature columns
"""

import os
import sys
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# make sure project root is importable for utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import MODEL_DIR, PROCESSED_DATA_PATH
from utils.logger import get_logger

log = get_logger("streamlit_app")

# --- Helpers: load artifacts & data -------------------------------------------------
@st.cache_resource
def load_artifacts():
    """
    Load trained model and preprocessor from MODEL_DIR.
    MODEL_DIR should point to folder that already contains:
      - rent_model.pkl
      - preprocessor.pkl
    """
    model_path = os.path.join(MODEL_DIR, "rent_model.pkl")
    preproc_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Preprocessor not found at {preproc_path}")

    log.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    log.info(f"Loading preprocessor from {preproc_path}")
    preprocessor = joblib.load(preproc_path)

    return model, preprocessor


@st.cache_data
def load_processed_df():
    """Load processed dataframe used to populate dropdowns and defaults."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df


def build_dropdown_options(df, categorical_cols):
    options = {}
    for c in categorical_cols:
        if c in df.columns:
            opts = sorted(df[c].dropna().astype(str).unique().tolist())
            options[c] = opts
        else:
            options[c] = []
    return options


def get_numerical_defaults(df, numerical_cols):
    defaults = {}
    for c in numerical_cols:
        if c in df.columns:
            # median is robust default
            defaults[c] = float(df[c].median()) if pd.api.types.is_numeric_dtype(df[c]) else 0.0
        else:
            defaults[c] = 0.0
    return defaults


# --- Align input with preprocessor expectations -------------------------------------
def build_input_dataframe(user_inputs: dict, preprocessor, processed_df):
    """
    Construct a DataFrame with the exact raw columns expected by the preprocessor.
    Strategy:
      - Preprocessor.transformers_ contains (name, transformer, column_list)
      - For any expected column missing in user_inputs, fill with default:
          * if categorical: most frequent value from processed_df (or 'Unknown')
          * if numeric: median from processed_df (or 0)
      - Return single-row DataFrame with columns in same order the preprocessor expects.
    """
    # Collect expected columns from the preprocessor
    expected_cols = []
    transformers = getattr(preprocessor, "transformers_", None)
    if transformers is None:
        # If preprocessor is not fitted, try to read feature_names_in_ (newer sklearn)
        # fallback: use keys present in user_inputs
        expected_cols = list(user_inputs.keys())
    else:
        for name, trans, cols in transformers:
            # ignore 'remainder' metadata
            if isinstance(cols, (list, tuple, np.ndarray)):
                expected_cols.extend(list(cols))

    # Prepare defaults from processed_df
    cat_defaults = {}
    num_defaults = {}
    for col in expected_cols:
        if col in processed_df.columns and processed_df[col].dtype == object:
            # most frequent as default
            vals = processed_df[col].dropna().astype(str)
            cat_defaults[col] = vals.mode().iloc[0] if not vals.mode().empty else "Unknown"
        elif col in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[col]):
            num_defaults[col] = float(processed_df[col].median())
        else:
            # unknown column: string default
            cat_defaults[col] = "Unknown"
            num_defaults[col] = 0.0

    # Build final single-row dict
    final = {}
    for col in expected_cols:
        if col in user_inputs and user_inputs[col] is not None:
            final[col] = user_inputs[col]
        else:
            # decide numeric or categorical
            if col in num_defaults:
                final[col] = num_defaults[col]
            else:
                final[col] = cat_defaults.get(col, "Unknown")

    # Return DataFrame
    df_final = pd.DataFrame([final], columns=expected_cols)
    return df_final


# --- Streamlit UI -----------------------------------------------------------------
st.set_page_config(page_title="Delhi Rent Prediction", layout="centered")
st.title("Delhi Rent Prediction")
st.markdown("Provide property details and get a predicted monthly rent. Model and preprocessing are reused from training artifacts.")

# Load artifacts and processed dataset
try:
    model, preprocessor = load_artifacts()
    processed_df = load_processed_df()
except Exception as e:
    st.error(f"Startup error: {e}")
    log.exception("startup error")
    st.stop()

# Identify categorical and numerical raw columns for UI (use preprocessor knowledge if available)
# Use preprocessor.transformers_ to extract columns lists
categorical_cols = []
numerical_cols = []
transformers = getattr(preprocessor, "transformers_", None)
if transformers:
    for name, trans, cols in transformers:
        if name.lower().startswith("categor"):
            categorical_cols.extend(list(cols))
        elif name.lower().startswith("numer"):
            numerical_cols.extend(list(cols))
        else:
            # Heuristic: if transformer contains OneHotEncoder it's categorical
            tname = str(type(trans)).lower()
            if "onehot" in tname:
                categorical_cols.extend(list(cols))
            else:
                numerical_cols.extend(list(cols))
else:
    # fallback: infer from processed_df
    categorical_cols = processed_df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = processed_df.select_dtypes(include=["int64", "float64", "float"]).columns.tolist()

# Build UI dropdown options and numeric defaults
dropdown_options = build_dropdown_options(processed_df, categorical_cols)
numeric_defaults = get_numerical_defaults(processed_df, numerical_cols)

# Layout: left column for inputs, right column for meta/defaults
with st.form("input_form"):
    st.subheader("Property details")
    inputs = {}

    # Categorical inputs (dropdown)
    for c in categorical_cols:
        opts = dropdown_options.get(c, [])
        if opts:
            inputs[c] = st.selectbox(f"{c}", options=opts, index=0)
        else:
            # free-text fallback
            inputs[c] = st.text_input(f"{c}", value="Unknown")

    # Numerical inputs
    for c in numerical_cols:
        default_val = numeric_defaults.get(c, 0.0)
        # Use number_input (floats allowed)
        inputs[c] = st.number_input(f"{c}", value=float(default_val))

    submitted = st.form_submit_button("Predict Rent")

if submitted:
    try:
        # Build DataFrame aligned to preprocessor expectation
        input_df = build_input_dataframe(inputs, preprocessor, processed_df)
        # Preprocess & predict
        X_proc = preprocessor.transform(input_df)
        pred = model.predict(X_proc)[0]
        st.metric(label="Predicted Monthly Rent (INR)", value=f"₹{pred:,.0f}")
        st.success("Prediction complete. See input summary below.")
        st.write("Input (aligned):")
        st.dataframe(input_df.T.rename(columns={0: "value"}))
    except Exception as ex:
        st.error(f"Prediction failed: {ex}")
        log.exception("prediction failed")

# Footer: help & links
st.markdown("---")
st.caption("Model artifacts loaded from `models/saved_models`. Processed dataset used for dropdowns: `data/processed/processed_rentals.csv`.")
