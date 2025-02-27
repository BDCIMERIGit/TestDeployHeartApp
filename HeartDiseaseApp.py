#Heart Disease App

import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('HeartModel.pkl')

# Streamlit app
st.title("Heart Disease Prediction App")

# User input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.slider("Chest Pain Type (CP)", 0, 3, 1)
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=200, value=120)
chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.slider("Resting Electrocardiographic Results (restecg)", 0, 2, 1)
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.slider("Slope of the Peak Exercise ST Segment", 0, 2, 1)
ca = st.slider("Number of Major Vessels (0-3) Colored by Fluoroscopy", 0, 3, 0)
thal = st.slider("Thalassemia Type (thal)", 0, 2, 1)

# Predict button
if st.button("Predict Heart Disease"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)[0]
    result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
    st.write(f"### Prediction: {result}")
