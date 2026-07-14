import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("wine_quality_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Wine Quality Prediction")
st.write("Enter the wine properties below and click Predict.")

fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
citric_acid = st.number_input("Citric Acid", value=0.00)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978)
ph = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "fixed acidity": [fixed_acidity],
        "volatile acidity": [volatile_acidity],
        "citric acid": [citric_acid],
        "residual sugar": [residual_sugar],
        "chlorides": [chlorides],
        "free sulfur dioxide": [free_sulfur_dioxide],
        "total sulfur dioxide": [total_sulfur_dioxide],
        "density": [density],
        "pH": [ph],
        "sulphates": [sulphates],
        "alcohol": [alcohol]
    })

    prediction = model.predict(input_data)

    st.subheader("Prediction")
    st.success(f"Predicted Wine Quality: {prediction[0]}")