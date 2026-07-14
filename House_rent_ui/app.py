import os
import joblib
import pandas as pd
import streamlit as st

# ==========================================================
# Load Model, Encoder and Scaler from Current Directory
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
artifacts = joblib.load(os.path.join(BASE_DIR, "house_rent_prediction.pkl"))
model = artifacts["model"]
encoder = artifacts["encoder"]
scaler = artifacts["scaler"]

# ==========================================================
# Streamlit Page Configuration
# ==========================================================

st.set_page_config(
    page_title="House Rent Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Rent Prediction System by Warisha")

st.write("Welcome! Enter the details below to predict the monthly house rent.")

# ==========================================================
# User Inputs
# ==========================================================

bhk = st.number_input(
    "BHK",
    min_value=1,
    max_value=10,
    value=2
)

size = st.number_input(
    "Size (Square Feet)",
    min_value=100,
    value=1000
)

area_type = st.selectbox(
    "Area Type",
    [
        "Super Area",
        "Carpet Area",
        "Built Area"
    ]
)

city = st.selectbox(
    "City",
    [
        "Mumbai",
        "Delhi",
        "Bangalore",
        "Chennai",
        "Hyderabad",
        "Kolkata"
    ]
)

furnishing_status = st.selectbox(
    "Furnishing Status",
    [
        "Unfurnished",
        "Semi-Furnished",
        "Furnished"
    ]
)

tenant_preferred = st.selectbox(
    "Tenant Preferred",
    [
        "Bachelors",
        "Family",
        "Bachelors/Family"
    ]
)

bathroom = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

point_of_contact = st.selectbox(
    "Point of Contact",
    [
        "Contact Owner",
        "Contact Agent",
        "Contact Builder"
    ]
)

# ==========================================================
# Prediction
# ==========================================================

if st.button("Predict Rent"):

    input_df = pd.DataFrame({
        "BHK": [bhk],
        "Size": [size],
        "Area Type": [area_type],
        "City": [city],
        "Furnishing Status": [furnishing_status],
        "Tenant Preferred": [tenant_preferred],
        "Bathroom": [bathroom],
        "Point of Contact": [point_of_contact]
    })

    # Encode categorical columns
    input_encoded = encoder.transform(input_df)

    # Scale features
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict(input_scaled)

    rent = prediction[0]

    st.success(f"🏠 Predicted Monthly Rent: ₹ {rent:,.2f}")
