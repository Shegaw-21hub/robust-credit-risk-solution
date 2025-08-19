import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn

# ------------------------------
# Load Data
# ------------------------------
data_path = '../data/processed/model_ready_data.csv'
data = pd.read_csv(data_path, encoding='utf-8-sig')

# Strip any whitespace from column names
data.columns = data.columns.str.strip()

target_col = 'target__is_high_risk'

# Drop rows where target is NaN
data = data.dropna(subset=[target_col])

X = data.drop(columns=[target_col])
y = data[target_col]

print(f"Loaded data with {X.shape[0]} rows and {X.shape[1]} features. Target column: '{target_col}'.")

# ------------------------------
# Train/Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Data split into {X_train.shape[0]} training and {X_test.shape[0]} testing samples.")

# ------------------------------
# Optional Preprocessing Pipeline
# ------------------------------
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ------------------------------
# Define Models and Parameter Grids
# ------------------------------
models_to_train = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'model__C': [0.01, 0.1, 1, 10],
            'model__penalty': ['l2'],
            'model__solver': ['liblinear']
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),
        'params': {
            'model__n_estimators': [50, 100],
            'model__max_depth': [None, 5, 10],
            'model__min_samples_split': [2, 5]
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(),
        'params': {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5]
        }
    }
}

# ------------------------------
# MLflow Setup (Windows Safe)
# ------------------------------
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri(os.path.abspath("mlruns"))
mlflow.set_experiment("credit_risk_modeling")

# ------------------------------
# Training Loop
# ------------------------------
best_model_overall = {'model_name': None, 'estimator': None, 'roc_auc': 0}

for model_name, config in models_to_train.items():
    print(f"\n--- Training {model_name} ---")
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', config['model'])
    ])
    
    grid = GridSearchCV(
        pipeline,
        config['params'],
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    
    best_pipeline = grid.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Best Params: {grid.best_params_}")
    print(f"ROC AUC: {roc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    mlflow.sklearn.log_model(best_pipeline, artifact_path=model_name)
    
    if roc > best_model_overall['roc_auc']:
        best_model_overall['model_name'] = model_name
        best_model_overall['estimator'] = best_pipeline
        best_model_overall['roc_auc'] = roc

# ------------------------------
# Save Artifacts
# ------------------------------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(best_model_overall['estimator'], 'artifacts/model.pkl')
joblib.dump(pipeline, 'artifacts/pipeline.pkl')  # Save preprocessing pipeline

print(f"\nBest overall model '{best_model_overall['model_name']}' saved locally and logged in MLflow.")
print(f"Best overall model ROC AUC: {best_model_overall['roc_auc']:.4f}")
