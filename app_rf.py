import streamlit as st
import numpy as np
import joblib
import pandas as pd

# --------------------------------------
# Load the Random Forest model and scaler using absolute paths
# --------------------------------------
model_path = r"C:\Users\mohdq\OneDrive\Desktop\internship projects\Mobile pricing.pro\random_forest_model.pkl"
scaler_path = r"C:\Users\mohdq\OneDrive\Desktop\internship projects\Mobile pricing.pro\scaler.pkl"

with open(model_path, "rb") as file:
    model = joblib.load(file)
with open(scaler_path, "rb") as file:
    scaler = joblib.load(file)

# --------------------------------------
# Streamlit UI Setup
# --------------------------------------
st.title("📱 Mobile Price Classification (Random Forest)")

# Define feature inputs using dropdowns for better usability
battery_power = st.selectbox("🔋 Battery Power (mAh)", 
                               [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000], 
                               index=3)
ram = st.selectbox("⚡ RAM (MB)", 
                   [256, 512, 1024, 2048, 3072, 4096, 6144, 8192], 
                   index=4)

# Display Quality as a single dropdown with human-friendly labels
display_quality = st.selectbox("🖥️ Display Quality", 
                               ["SD (Low)", "HD (Standard)", "HD+ (Better)", "4K (Best)"], 
                               index=1)
display_mapping = {"SD (Low)": 0, "HD (Standard)": 1, "HD+ (Better)": 2, "4K (Best)": 3}
display_quality = display_mapping[display_quality]

# Screen Size (in inches)
sc_h = st.selectbox("📏 Screen Size", 
                    ["4 inches (Small)", "5 inches (Medium)", "6 inches (Large)", "7 inches (Very Large)"], 
                    index=1)
screen_size_mapping = {
    "4 inches (Small)": 4,
    "5 inches (Medium)": 5,
    "6 inches (Large)": 6,
    "7 inches (Very Large)": 7
}
sc_h = screen_size_mapping[sc_h]

talk_time = st.selectbox("⏳ Talk Time (hours)", list(range(2, 21)), index=8)
m_dep = st.selectbox("📏 Phone Thickness (cm)", [0.1, 0.3, 0.5, 0.7, 1.0], index=2)
n_cores = st.selectbox("🖥️ Number of Processor Cores", [1, 2, 4, 6, 8], index=2)
screen_to_body = st.slider("📱 Screen to Body Ratio (%)", min_value=10.0, max_value=90.0, step=0.5, value=50.0)

# Categorical inputs with Yes/No options
blue = st.radio("📶 Bluetooth", ["No", "Yes"])
four_g = st.radio("📡 4G", ["No", "Yes"])
three_g = st.radio("📶 3G", ["No", "Yes"])
wifi = st.radio("📡 WiFi", ["No", "Yes"])

# Convert "Yes"/"No" inputs to 1/0
blue = 1 if blue == "Yes" else 0
four_g = 1 if four_g == "Yes" else 0
three_g = 1 if three_g == "Yes" else 0
wifi = 1 if wifi == "Yes" else 0

# --------------------------------------
# Prepare Input Data
# --------------------------------------
# Feature order must match training order:
# ["battery_power", "blue", "four_g", "m_dep", "n_cores", "ram", "sc_h", "talk_time", "three_g", "wifi", "screen_to_body", "display_quality"]
input_df = pd.DataFrame([[
    battery_power, blue, four_g, m_dep, n_cores, ram,
    sc_h, talk_time, three_g, wifi, screen_to_body, display_quality
]], columns=[
    "battery_power", "blue", "four_g", "m_dep", "n_cores", "ram",
    "sc_h", "talk_time", "three_g", "wifi", "screen_to_body", "display_quality"
])

st.write("📌 Input Features:", input_df)

# --------------------------------------
# Prediction
# --------------------------------------
if st.button("🔍 Predict Price Range"):
    # Scale the input features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Define readable price categories
    price_labels = {
        0: "Low-cost 📉 (Entry-level, basic features)",
        1: "Budget 💰 (Affordable, decent performance)",
        2: "Mid-range 📱 (Good performance, reasonable price)",
        3: "Premium 🔥 (High-end, flagship level)"
    }
    
    # Display prediction
    st.success(f"💰 **Predicted Price Range:** {price_labels[prediction]}")

