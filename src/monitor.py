import pandas as pd
import joblib
from evidently import Report
from evidently.presets import DataDriftPreset, RegressionPreset
import os
from datetime import datetime
import shutil

def run_monitoring_report():
    """
    Generates data drift and model performance reports with timestamped filenames
    and also updates a 'latest' copy.
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

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------- Data Drift Report ----------------
    print("Generating Data Drift Report...")
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=ref_data, current_data=current_data)

    drift_filename = f"reports/data_drift_report_{timestamp}.html"
    data_drift_report.save_html(drift_filename)

    # Also update 'latest'
    shutil.copy(drift_filename, "reports/data_drift_report_latest.html")

    print(f"Data drift report saved to {drift_filename}")

    # ---------------- Model Performance Report ----------------
    print("Generating Model Performance Report...")

    preprocessed_current = pipeline.transform(current_data.drop(columns=['target__is_high_risk'], errors='ignore'))
    current_data['prediction'] = model.predict(preprocessed_current)

    model_performance_report = Report(metrics=[RegressionPreset()])
    model_performance_report.run(
        reference_data=ref_data,
        current_data=current_data,
        column_mapping=None
    )

    perf_filename = f"reports/model_performance_report_{timestamp}.html"
    model_performance_report.save_html(perf_filename)

    # Also update 'latest'
    shutil.copy(perf_filename, "reports/model_performance_report_latest.html")

    print(f"Model performance report saved to {perf_filename}")

if __name__ == "__main__":
    run_monitoring_report()
