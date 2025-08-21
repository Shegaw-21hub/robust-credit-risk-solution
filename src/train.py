import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

# Paths
DATA_PATH = "data/processed/transactions_processed.csv"
MODEL_SAVE_PATH = "models/rf_model.pkl"

# Load data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Data loaded. Shape: {df.shape}")

# Determine target column
if 'is_high_risk' in df.columns:
    target_col = 'is_high_risk'
elif 'risk_level' in df.columns:
    df['is_high_risk'] = df['risk_level'].map({'low': 0, 'high': 1})
    target_col = 'is_high_risk'
elif 'FraudResult' in df.columns:
    df['is_high_risk'] = df['FraudResult']  # Use FraudResult as target
    target_col = 'is_high_risk'
else:
    raise KeyError("Cannot proceed without 'is_high_risk', 'risk_level', or 'FraudResult' as target.")

print(f"Using target column: {target_col}")

# Drop non-numeric or ID columns
non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
id_like_cols = [c for c in df.columns if 'id' in c.lower()]
drop_cols = list(set(non_numeric_cols + id_like_cols) - set([target_col]))
df = df.drop(columns=drop_cols)
print(f"Dropped columns: {drop_cols}")

# Features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train RandomForest with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
print("Training RandomForest...")
grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")

# Evaluate
y_pred = grid.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
import joblib
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(grid.best_estimator_, MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
