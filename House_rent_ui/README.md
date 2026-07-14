# House Rent Prediction System by Warisha

A Streamlit web application that predicts monthly house rent based on various property features.

## Features
- Predict house rent using a trained Linear Regression model
- User-friendly interface with input fields for property details
- Real-time predictions

## Project Structure
```
House_rent_ui/
├── app.py                      # Main Streamlit application
├── model_training.ipynb        # Jupyter notebook for model training
├── house_rent_prediction.pkl   # Trained model artifacts (model, encoder, scaler)
├── House_Rent_Dataset.csv      # Dataset used for training
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation
1. Clone or download the project
2. Create a virtual environment (optional but recommended)
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and go to http://localhost:8501
3. Enter the property details and click "Predict Rent"

## Model Details
- Algorithm: Linear Regression
- Features used:
  - BHK
  - Size (Square Feet)
  - Area Type
  - City
  - Furnishing Status
  - Tenant Preferred
  - Bathroom
  - Point of Contact
