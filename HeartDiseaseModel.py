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
