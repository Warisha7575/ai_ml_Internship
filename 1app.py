import joblib
import pandas as pd
# Load Pipeline
loaded_pipeline = joblib.load("house_rent_pipeline.pkl")
# Predict a New House
new_house = pd.DataFrame({
 "BHK": [2],
 "Size": [1200],
 "Area Type": ["Super Area"],
 "City": ["Delhi"],
 "Furnishing Status": ["Semi-Furnished"],
 "Tenant Preferred": ["Family"],
 "Bathroom": [2],
 "Point of Contact": ["Contact Owner"]
})
prediction = loaded_pipeline.predict(new_house)
print(f"Predicted Rent: ₹{prediction[0]:,.0f}")
