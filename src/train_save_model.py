# src/train_save_model.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
df = pd.read_csv("data/your_dataset.csv")  # replace with your data path

X = df[['feature1', 'feature2', 'feature3']]  # replace with your features
y = df['target']  # replace with your target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = Pipeline([
    ('scaler', StandardScaler())
])

X_train_scaled = preprocessor.fit_transform(X_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and pipeline
os.makedirs("models", exist_ok=True)
joblib.dump(preprocessor, "models/feature_pipeline.pkl")
joblib.dump(model, "models/rf_model.pkl")

print("Model and pipeline saved successfully!")
