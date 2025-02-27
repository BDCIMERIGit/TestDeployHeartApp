#Test Deploy Heart Disease App Streamlit

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import requests

# Load the dataset from GitHub
url = 'https://raw.githubusercontent.com/BDCIMERIGit/TestDeployHeartApp/main/heart.csv'  # Update this with the actual URL
response = requests.get(url)
with open('heart.csv', 'wb') as f:
    f.write(response.content)

df = pd.read_csv('heart.csv')

# Split into features (X) and target (y)
X = df.drop(columns=['target'])  # Assuming 'target' is the label column
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline with SVM model
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(C=1, gamma='scale', kernel='linear'))
])

# Train the model
svm_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = svm_pipeline.predict(X_test)

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model
joblib.dump(svm_pipeline, 'HeartModel.pkl')
print("Model saved as 'HeartModel.pkl'")

#Joblib error fix
#open github codespaces

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
