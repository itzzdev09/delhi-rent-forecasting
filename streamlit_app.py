import os
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ----------------------------
# Load Models & Preprocessor
# ----------------------------
MODEL_DIR = os.path.join("models", "saved_models")
model_path = os.path.join(MODEL_DIR, "rent_model.pkl")
preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
    st.error("❌ Model or preprocessor files not found. Check your paths.")
else:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

# ----------------------------
# Streamlit UI Layout
# ----------------------------
st.set_page_config(page_title="Delhi Rent Forecasting", layout="wide")

st.title("🏠 Delhi Rent Forecasting Dashboard")

# Sidebar Inputs
st.sidebar.header("Input Features")

area = st.sidebar.number_input("Area (Sq. Ft.)", min_value=100, max_value=5000, step=50, value=1000)
bhk = st.sidebar.radio("Bedrooms (BHK)", [1, 2, 3, "4+"])
bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 5, 2)
balconies = st.sidebar.slider("Number of Balconies", 0, 3, 1)

location = st.sidebar.selectbox("Location", ["Dwarka", "Vasant Kunj", "Rohini"])

# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("Predict Rent"):
    # Prepare input
    input_df = pd.DataFrame([{
        "house_type": "Apartment",
        "house_size": f"{bhk} BHK" if bhk != "4+" else "4 BHK",
        "location": location,
        "latitude": 28.61,   # you can map location → lat/lon from your dataset
        "longitude": 77.04,
        "numBathrooms": bathrooms,
        "numBalconies": balconies,
        "area": area
    }])

    X_processed = preprocessor.transform(input_df)
    prediction = model.predict(X_processed)[0]

    # ----------------------------
    # Forecasted Rent Card
    # ----------------------------
    st.markdown(
        f"""
        <div style="text-align:center; background-color:#f8f9fa; padding:20px; 
        border-radius:15px; font-size:28px; font-weight:bold;">
        Forecasted Rent: ₹ {prediction:,.0f}
        </div>
        """, unsafe_allow_html=True
    )

    # ----------------------------
    # Map Visualization
    # ----------------------------
    st.subheader("📍 Location Map")
    map_df = pd.DataFrame([{
        "lat": input_df["latitude"][0],
        "lon": input_df["longitude"][0],
        "location": location,
        "predicted_rent": prediction
    }])

    fig_map = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        color="predicted_rent",
        size=[15],
        hover_name="location",
        zoom=11,
        height=400
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

    # ----------------------------
    # Historical vs Predicted Trends (Dummy Data Example)
    # ----------------------------
    st.subheader("📊 Historical vs. Predicted Rent Trends")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    actual_rent = [20000, 22000, 21000, 23000, 25000, 24000]
    predicted_rent = [prediction * 0.9, prediction * 0.95, prediction,
                      prediction * 1.05, prediction * 1.1, prediction]

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=months, y=actual_rent,
                                  mode="lines+markers", name="Actual Rent"))
    fig_line.add_trace(go.Scatter(x=months, y=predicted_rent,
                                  mode="lines+markers", name="Predicted Rent"))

    st.plotly_chart(fig_line, use_container_width=True)
