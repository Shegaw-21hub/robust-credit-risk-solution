import pandas as pd
import joblib
from evidently import Report  # ✅ FIXED: Updated import
from evidently.presets import DataDriftPreset, RegressionPreset

import os

def run_monitoring_report():
    """
    Generates data drift and model performance reports.
    """
    raw_data_path = 'data/raw/data.csv'
    model_path = 'artifacts/model.pkl'
    pipeline_path = 'artifacts/pipeline.pkl'

    os.makedirs('reports', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)

    try:
        full_data = pd.read_csv(raw_data_path)
        model = joblib.load(model_path)
        pipeline = joblib.load(pipeline_path)

        ref_data = full_data.sample(frac=0.8, random_state=42)
        current_data = full_data.drop(ref_data.index)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your data and model files exist in the correct paths.")
        return

    # Generate Data Drift Report
    print("Generating Data Drift Report...")
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=ref_data, current_data=current_data)
    data_drift_report.save_html("reports/data_drift_report.html")
    print("Data drift report saved to reports/data_drift_report.html")

    # Generate Model Performance Report
    print("Generating Model Performance Report...")

    preprocessed_current = pipeline.transform(current_data)
    current_data['prediction'] = model.predict(preprocessed_current)  # ✅ Add predictions

    # Make sure your current_data has the actual target column, e.g., 'is_high_risk'
    model_performance_report = Report(metrics=[RegressionPreset()])
    model_performance_report.run(
        reference_data=ref_data,
        current_data=current_data,
        column_mapping=None
    )
    model_performance_report.save_html("reports/model_performance_report.html")
    print("Model performance report saved to reports/model_performance_report.html")

if __name__ == "__main__":
    run_monitoring_report()
