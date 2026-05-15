import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/xgb_burnout_model.pkl")

st.title("🧠 Tech Burnout Checker")
st.write("Predict burnout levels based on workplace and lifestyle factors.")

# Collect user inputs
age = st.number_input("Age", min_value=18, max_value=65, value=25)
work_hours = st.slider("Work Hours per Week", 20, 80, 40)
sleep_hours = st.slider("Sleep Hours per Night", 3, 10, 7)
stress_score = st.slider("Stress Score (1-10)", 1, 10, 5)
manager_support = st.slider("Manager Support (1-10)", 1, 10, 5)
job_satisfaction = st.slider("Job Satisfaction (1-10)", 1, 10, 6)

# Create dataframe for prediction
input_data = pd.DataFrame({
    "age": [age],
    "work_hours": [work_hours],
    "sleep_hours": [sleep_hours],
    "stress_score": [stress_score],
    "manager_support": [manager_support],
    "job_satisfaction": [job_satisfaction]
})

# Predict
prediction = model.predict(input_data)[0]
labels = {0: "Low Burnout", 1: "Moderate Burnout", 2: "High Burnout", 3: "Severe Burnout"}

st.subheader("Prediction")
st.success(f"Your predicted burnout level: **{labels[prediction]}**")
