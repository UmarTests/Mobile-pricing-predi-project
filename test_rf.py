import joblib
import numpy as np
import pandas as pd

# Load the Random Forest model and scaler
with open(r"C:\Users\mohdq\OneDrive\Desktop\internship projects\Mobile pricing.pro\random_forest_model.pkl", "rb") as file:
    model = joblib.load(file)

scaler = joblib.load("scaler.pkl")

# Define feature order (must match training features)
feature_columns = [
    "battery_power", "blue", "four_g", "m_dep", "n_cores", 
    "ram", "sc_h", "talk_time", "three_g", "wifi", 
    "screen_to_body", "display_quality"
]

# Define test cases (Low-end, Mid-range, High-end)
test_cases = [
    {
        "name": "Low-end Phone",
        "battery_power": 1000, "blue": 0, "four_g": 0, "m_dep": 0.5, "n_cores": 2,
        "ram": 512, "sc_h": 4, "talk_time": 5, "three_g": 0, "wifi": 0, 
        "screen_to_body": 30, "display_quality": 0  # SD
    },
    {
        "name": "Mid-range Phone (Adjusted)",
        "battery_power": 1800, "blue": 1, "four_g": 1, "m_dep": 0.6, "n_cores": 4,
        "ram": 1536, "sc_h": 5, "talk_time": 8, "three_g": 1, "wifi": 1, 
        "screen_to_body": 45, "display_quality": 1  # HD
    },
    {
        "name": "High-end Phone",
        "battery_power": 4000, "blue": 1, "four_g": 1, "m_dep": 0.7, "n_cores": 8,
        "ram": 8192, "sc_h": 7, "talk_time": 15, "three_g": 1, "wifi": 1, 
        "screen_to_body": 75, "display_quality": 3  # 4K
    }
]

for case in test_cases:
    input_df = pd.DataFrame([case], columns=feature_columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    price_labels = {0: "Low-cost 📉", 1: "Budget 💰", 2: "Mid-range 📱", 3: "Premium 🔥"}
    print(f"\n📌 {case['name']} Prediction: {price_labels[prediction]}")
