import streamlit as st
import pandas as pd
import joblib

# Load model and scaler (if you saved preprocessing)
model = joblib.load("xgb_burnout_model.pkl")
try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = None

# Feature names the model expects
FEATURES = ['age', 'work_hours', 'sleep_hours', 'stress_score',
            'manager_support', 'job_satisfaction']

# App UI
st.set_page_config(page_title="Tech Burnout Checker", page_icon="🧠", layout="centered")

st.title("🧠 Tech Burnout Checker")
st.markdown("""
Welcome to the **Tech Burnout Checker**.  
Provide your workplace and lifestyle details below, and the model will predict your burnout level.
""")

# Sidebar for inputs
st.sidebar.header("📋 Input Your Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=65, value=25)
work_hours = st.sidebar.slider("Work Hours per Week", 20, 80, 40)
sleep_hours = st.sidebar.slider("Sleep Hours per Night", 3, 10, 7)
stress_score = st.sidebar.slider("Stress Score (1-10)", 1, 10, 5)
manager_support = st.sidebar.slider("Manager Support (1-10)", 1, 10, 5)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-10)", 1, 10, 6)

# Build input DataFrame with correct feature names
input_data = pd.DataFrame([[age, work_hours, sleep_hours, stress_score,
                            manager_support, job_satisfaction]], columns=FEATURES)

# Apply scaler if available
if scaler:
    input_data = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data, validate_features=False)[0]
labels = {0: "Low Burnout", 1: "Moderate Burnout", 2: "High Burnout", 3: "Severe Burnout"}

# Display result
st.subheader("🔮 Prediction")
st.success(f"Your predicted burnout level: **{labels[prediction]}**")

# Optional: Add some style
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & XGBoost")
