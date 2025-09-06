import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

from utils.config import MODEL_DIR, PROCESSED_DATA_PATH

# Load model + preprocessor
model_path = os.path.join(MODEL_DIR, "saved_models", "rent_model.pkl")
preprocessor_path = os.path.join(MODEL_DIR, "saved_models", "preprocessor.pkl")

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# Load processed data (for dropdowns, trends, and map plotting)
df = pd.read_csv(PROCESSED_DATA_PATH)


# -------------------------------
# UI
# -------------------------------
st.set_page_config(
    page_title="Delhi Rent Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Delhi Rent Forecasting Dashboard")

# Sidebar Inputs
st.sidebar.header("Enter Property Details")

area = st.sidebar.number_input("Area (Sq. Ft.)", min_value=100, max_value=5000, value=1000, step=50)
bhk = st.sidebar.radio("Bedrooms (BHK)", [1, 2, 3, "4+"])
location = st.sidebar.selectbox("Location", sorted(df["location"].dropna().unique().tolist()))
bathroom = st.sidebar.slider("Number of Bathrooms", 1, 5, 2)
balconies = st.sidebar.slider("Number of Balconies", 0, 3, 1)

# Build input dict
input_data = {
    "house_type": "Apartment",
    "house_size": f"{bhk} BHK" if bhk != "4+" else "4 BHK",
    "location": location,
    "latitude": df[df["location"] == location]["latitude"].mean(),
    "longitude": df[df["location"] == location]["longitude"].mean(),
    "numBathrooms": bathroom,
    "numBalconies": balconies,
    "size": area
}

# Prediction
X_input = pd.DataFrame([input_data])
X_processed = preprocessor.transform(X_input)
pred = model.predict(X_processed)[0]

# Forecasted Rent Display
st.metric("Forecasted Rent", f"₹ {round(pred, 2):,.0f}")

# -------------------------------
# Map Visualization
# -------------------------------
st.subheader("📍 Location-wise Rent Map")

map_data = df.groupby("location").agg({
    "latitude": "mean",
    "longitude": "mean",
    "price": "mean"
}).reset_index()

fig_map = px.scatter_mapbox(
    map_data,
    lat="latitude",
    lon="longitude",
    color="price",
    size="price",
    hover_name="location",
    color_continuous_scale="YlOrRd",
    zoom=10,
    height=400
)
fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

# -------------------------------
# Historical vs Predicted Trends
# -------------------------------
st.subheader("📊 Historical vs. Predicted Rent Trends")

# Simulated monthly trend (replace with real historical data if available)
trend_data = pd.DataFrame({
    "Month": pd.date_range("2023-01-01", periods=12, freq="M"),
    "Actual Rent": (df["price"].mean() * (1 + 0.1 * pd.Series(range(12)))).values,
    "Predicted Rent": (df["price"].mean() * (1 + 0.08 * pd.Series(range(12)))).values
})

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=trend_data["Month"], y=trend_data["Actual Rent"],
                               mode="lines+markers", name="Actual Rent", line=dict(color="green")))
fig_trend.add_trace(go.Scatter(x=trend_data["Month"], y=trend_data["Predicted Rent"],
                               mode="lines+markers", name="Predicted Rent", line=dict(color="red")))
fig_trend.update_layout(xaxis_title="Month", yaxis_title="Rent (₹)", height=400)

st.plotly_chart(fig_trend, use_container_width=True)
