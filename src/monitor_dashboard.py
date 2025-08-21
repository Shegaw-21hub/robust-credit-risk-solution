import streamlit as st
import pandas as pd
import joblib
from evidently import Report
from evidently.presets import DataDriftPreset, RegressionPreset
import os

# ---------------------------
# File paths
# ---------------------------
RAW_DATA_PATH = 'data/raw/data.csv'
MODEL_PATH = 'artifacts/model.pkl'
PIPELINE_PATH = 'artifacts/pipeline.pkl'

# ---------------------------
# Load model and pipeline
# ---------------------------
@st.cache_resource
def load_model_and_pipeline():
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline

model, pipeline = load_model_and_pipeline()

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv(RAW_DATA_PATH)

data = load_data()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Credit Risk Monitoring Dashboard")
st.write("Interactive monitoring for model performance and data drift.")

# Tabs for different views
tab1, tab2 = st.tabs(["Data Drift", "Model Performance"])

with tab1:
    st.header("Data Drift Report")
    st.write("Reference data: 80% sample, Current data: remaining 20%")
    
    ref_data = data.sample(frac=0.8, random_state=42)
    current_data = data.drop(ref_data.index)
    
    if st.button("Generate Data Drift Report"):
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=ref_data, current_data=current_data)
        drift_report.save_html("reports/data_drift_report.html")
        st.success("Data drift report saved to reports/data_drift_report.html")
        st.components.v1.html(open("reports/data_drift_report.html", 'r').read(), height=600, scrolling=True)

with tab2:
    st.header("Model Performance Report")
    current_data_copy = current_data.copy()
    preprocessed = pipeline.transform(current_data_copy)
    current_data_copy['prediction'] = model.predict(preprocessed)
    
    if st.button("Generate Model Performance Report"):
        perf_report = Report(metrics=[RegressionPreset()])
        perf_report.run(reference_data=ref_data, current_data=current_data_copy)
        perf_report.save_html("reports/model_performance_report.html")
        st.success("Model performance report saved to reports/model_performance_report.html")
        st.components.v1.html(open("reports/model_performance_report.html", 'r').read(), height=600, scrolling=True)
