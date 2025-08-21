# Part 1: Title, and Executive Summary
This section provides a high-level overview, immediately showcasing the project's purpose and key technologies.
## Credit Risk Modeling Project: An End-to-End MLOps Approach

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10.1-orange.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)
[![GitHub Actions CI](https://github.com/your-username/credit-risk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/credit-risk-model/actions/workflows/ci.yml)
This repository presents a robust and production-ready solution for credit risk assessment, encompassing the entire machine learning lifecycle from initial data understanding and sophisticated preprocessing to advanced model training, rigorous evaluation, and streamlined experiment tracking with MLflow. It integrates modern MLOps practices, including containerization with Docker and automated Continuous Integration (CI) with GitHub Actions, ensuring reproducibility, scalability, and maintainability.

The core objective is to predict customer creditworthiness based on transactional behavior, providing an essential tool for financial institutions to manage risk, optimize lending decisions, and ensure regulatory compliance.

---
# Part 2: Credit Scoring Business Understanding & Regulatory Compliance

## 🎯 Credit Scoring Business Understanding

Credit scoring is a cornerstone of responsible lending, enabling financial institutions to quantify and manage the risk associated with extending credit. Our approach is guided by industry best practices and regulatory considerations:

### 1. Basel II Accord and Model Interpretability

The Basel II Accord serves as a foundational framework for banking regulation, emphasizing three pillars crucial for risk management: Minimum Capital Requirements, Supervisory Review, and Market Discipline. For our credit risk model, adherence to these principles is paramount:

* **Regulatory Compliance**: The Accord mandates that banks possess demonstrably sound and empirically validated risk measurement systems. This necessitates the development of an **interpretable model**, where the underlying decision-making logic is transparent and explicable to regulatory bodies. This model is designed to facilitate clear explanations of how a credit decision is reached.
* **Comprehensive Documentation**: Rigorous documentation of every stage of our modeling process is essential. This includes the rationale behind feature selection, the methodologies for creating proxy variables, and the detailed procedures for model validation and backtesting.
* **Granular Risk Sensitivity**: The model is engineered to accurately differentiate between various levels of credit risk. This granular understanding is critical for ensuring that capital allocation is precisely aligned with the risk weights associated with different loan portfolios, thereby optimizing capital efficiency.

### 2. Proxy Variable Necessity and Associated Risks

In scenarios where direct default data is unavailable or scarce (e.g., for new customer segments or novel product offerings), the use of proxy variables becomes a necessity to approximate credit risk.

* **Necessity**: Our model utilizes a proxy based on RFM (Recency, Frequency, Monetary) metrics derived from transactional behavior. This approach allows us to infer patterns correlating with repayment likelihood, bridging the gap created by the absence of direct default labels.
* **Business Risks**:
    * **Misclassification Risk**: A poorly constructed or applied proxy can lead to significant misclassification, resulting in either:
        * **False Positives**: Mislabeling creditworthy customers as high-risk, leading to lost revenue opportunities.
        * **False Negatives**: Mislabeling high-risk customers as low-risk, leading to increased default rates and financial losses.
    * **Concept Drift & Data Fidelity**: Behavioral patterns observed in e-commerce or other transactional data may not perfectly or perpetually correlate with traditional credit repayment behavior. This introduces a risk of concept drift, where the relationship between the proxy and actual credit risk evolves over time.
    * **Regulatory Scrutiny**: The methodology underpinning any proxy variable must be meticulously justified and validated to satisfy stringent compliance requirements and withstand regulatory audits.

### 3. Model Complexity Trade-offs

Choosing the right model complexity involves a critical balance between predictive power, interpretability, and regulatory acceptance:

* **Simple Models (e.g., Logistic Regression with WoE Transformation)**:
    * *Advantages*: Highly interpretable ("white box" nature), directly compliant with "right to explanation" regulations (e.g., GDPR), simpler to validate, audit, and deploy. They offer transparency crucial for financial applications.
    * *Disadvantages*: May struggle to capture highly complex, non-linear relationships within the data, potentially leading to sub-optimal predictive power compared to more sophisticated models.

* **Complex Models (e.g., Gradient Boosting Machines like LightGBM, XGBoost)**:
    * *Advantages*: Generally offer superior predictive accuracy, capable of capturing intricate feature interactions and non-linear patterns that simpler models might miss.
    * *Disadvantages*: Their "black box" nature can raise significant regulatory concerns, making it challenging to explain individual credit decisions to customers or regulators, which is a key requirement in consumer finance.

**Recommended Approach**: Our strategy prioritizes interpretability and regulatory alignment. We advocate starting with more interpretable models (e.g., Logistic Regression with WoE) to establish a strong, explainable baseline. Increased model complexity (e.g., moving to Gradient Boosting) should only be pursued if demonstrably justified by substantial and consistent performance gains that convincingly outweigh the added regulatory scrutiny and operational complexities.

---
# Part 3: Project Architecture & Structure
This section outlines the project's directory structure, updated to include all the new components like Dockerfiles, CI, and the refactored API.
## 🏛️ Project Architecture and Structure

The project is meticulously organized into logical directories, adhering to best practices for MLOps and software engineering, ensuring maintainability, clarity, and scalability.
```
credit-risk-model/
├── .github/                            # GitHub Actions workflows for CI/CD
│   └── workflows/
│       └── ci.yml                      # Automated Continuous Integration pipeline
├── data/
│   ├── processed/
│   │   ├── eda_plots/                  # Directory for EDA visualizations
│   │   │   └── * # Placeholder for EDA plot files
│   │   └── model_ready_data.csv        # Cleaned, processed, and feature-engineered data (output of data_processing.py)
│   └── raw/
│       └── [raw_data_files]            # Original, raw dataset files
├── models/
│   └── preprocessor.pkl                # Saved artifact of the fitted scikit-learn preprocessor
│   └── best_model.pkl                  # Saved artifact of the best trained model (for local inference/backup)
├── notebooks/
│   └── 1.0-eda.ipynb                   # Jupyter Notebook for Exploratory Data Analysis
├── src/
│   ├── api/                            # FastAPI application for model serving
│   │   ├── init.py                 # Python package initializer
│   │   ├── main.py                     # Main FastAPI application entry point
│   │   └── pydantic_models.py          # Pydantic data models for API requests/responses
│   ├── data_processing.py              # Script for data cleaning and feature engineering
│   └── train.py                        # Script for model training, evaluation, and MLflow tracking
│   │── monitor.py                  
│   ├── init.py                     # Python package initializer
│   └── test_data_processing.py         # Unit tests for the data_processing.py script
├── .dockerignore                       # Specifies files/directories to exclude from Docker image builds
├── .gitignore                          # Specifies intentionally untracked files to ignore
├── Dockerfile                          # Dockerfile for building the FastAPI application image
├── docker-compose.yml                  # Docker Compose configuration for multi-service local deployment (API + MLflow)
├── mlruns/                             # MLflow local tracking directory (generated by train.py, persisted by Docker Compose)
├── README.md                           # Project overview and documentation (this file)
└── requirements.txt                    # Python dependencies for the project
```
# Part 4: Setup and Running the Project (Local Development with Docker Compose)
This section provides clear, step-by-step instructions for setting up and running the entire project locally using Docker Compose, emphasizing ease of use.
## 🚀 Setup and Running the Project

The most straightforward way to set up and run this project for development, including both the FastAPI application and the MLflow tracking server, is using Docker Compose.

### Prerequisites

* **Docker Desktop (or Docker Engine)**: Ensure Docker is installed and running on your system (Windows, macOS, or Linux).
* **Git**: For cloning the repository.
* **Python 3.8+ & pip**: While Docker handles most dependencies, you might need a local Python environment for initial setup or running specific scripts outside Docker.

### 1. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone https://github.com/Shegaw-21hub/credit-risk-model
cd credit-risk-model
```
## 2. Set Up Python Virtual Environment (Recommended for Local Scripts)
While Docker Compose handles the main application environment, it's good practice to set up a local virtual environment for running scripts like data_processing.py or train.py directly, or for local development tools.
```
python -m venv venv
# On Windows
.\venv\Scripts\activate
```
## 3. Install Python Dependencies
Install all required Python packages into your active virtual environment:
```
pip install -r requirements.txt
```
## 4. Data Acquisition (Manual Step)
Ensure your raw dataset files are placed within the data/raw/ directory according to project specifications. This step is usually manual, depending on where your raw data originates.

## 5. Run the Data Processing Pipeline
This script cleans the raw data, performs feature engineering (including RFM and WoE transfor
mations), and generates the model_ready_data.csv file.
```
python src/data_processing.py
```
## 6. Run Data Processing Tests
To ensure the robustness and correctness of the data processing pipeline, execute its unit tests:
```
python src/tests/test_data_processing.py
```
## python src/tests/test_data_processing.py
This script orchestrates the model training process. It trains various machine learning models, performs hyperparameter tuning, evaluates their performance, and logs all results comprehensively using MLflow. The best model is also registered in the MLflow Model Registry.
```
python src/train.py
```
## 8. Run the Project with Docker Compose
This is the most efficient way to run both your FastAPI application and the MLflow Tracking Server concurrently.
```
# From the project root directory
docker-compose up --build
```
This command will:

Build the web service's Docker image (your FastAPI application) based on the Dockerfile.

Start both the web service (FastAPI) and the mlflow service (MLflow Tracking Server).

Mount your local project directory into the web container, allowing for live code changes during development (Uvicorn's --reload takes advantage of this).

Persist your MLflow experiments in the mlruns/ directory on your host machine.

Accessing Services:
MLflow UI (Experiment Tracking): Open your web browser and navigate to http://localhost:5000

FastAPI Application (API Documentation): Open your web browser and navigate to http://localhost:8000/docs
## 9. Monitoring & Drift Detection**  
  - `monitor.py` → Generates **Evidently reports** (data drift + performance).  
  - `monitor_dashboard.py` → **Streamlit dashboard** for interactive monitoring.  
  - `monitor.yml` → **GitHub Actions workflow** to run monitoring daily and archive reports.





