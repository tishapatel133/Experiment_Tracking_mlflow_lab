# Credit Card Default Prediction Lab

A comprehensive end-to-end machine learning lab demonstrating best practices for credit card default prediction using Python, scikit-learn, and MLflow for experiment tracking and model deployment.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

This lab provides a complete workflow for building, tracking, and deploying a machine learning model to predict credit card payment defaults. The project demonstrates:

- **Data Engineering**: Feature preprocessing with scikit-learn pipelines
- **Model Development**: Baseline and tuned models with hyperparameter optimization
- **Experiment Tracking**: MLflow integration for reproducible experiments
- **Model Registry**: Version control and stage transitions (Production/Staging)
- **Model Serving**: Both batch and real-time inference capabilities
- **Evaluation**: Comprehensive metrics including AUC, PR-AUC, F1, and confusion matrices

---

## Features

- **End-to-End ML Pipeline**: From raw data to production deployment
- **Experiment Tracking**: Automatic logging of parameters, metrics, and artifacts
- **Model Versioning**: Registry with stage management (Production/Staging)
- **Dual Inference Modes**: Batch scoring and REST API serving
- **Class Imbalance Handling**: Built-in support for imbalanced datasets
- **Feature Engineering**: Automated preprocessing with ColumnTransformer
- **Model Interpretability**: Feature importance analysis

---

## Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **IDE**: VS Code, Jupyter Notebook, or JupyterLab
- **OS**: Windows, macOS, or Linux

### Python Packages
```bash
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
mlflow>=2.0.0
cloudpickle>=2.0.0
ipykernel>=6.0.0
xlrd==1.2.0  # For reading .xls files
```

---

## Installation

### 1. Extract the ZIP File
```bash
unzip credit-card-default-lab.zip
cd credit-card-default-lab
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv credit_env

# Activate virtual environment
# On Windows:
credit_env\Scripts\activate
# On macOS/Linux:
source credit_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn mlflow cloudpickle ipykernel xlrd==1.2.0
```

### 4. Set Up MLflow Tracking Server
In a separate terminal window:
```bash
# Activate the same virtual environment
source credit_env/bin/activate  # or credit_env\Scripts\activate on Windows

# Start MLflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

The MLflow UI will be available at: `http://127.0.0.1:5000`

---

## Dataset

### Source
**UCI Credit Card Default Dataset**
- **Records**: 30,000 credit card customers
- **Features**: 23 input features
- **Target**: Binary classification (default/no default)
- **Format**: Excel (.xls) or CSV

### Data Dictionary

| Feature Type | Features | Description |
|-------------|----------|-------------|
| **Demographic** | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Customer demographics |
| **Financial** | `LIMIT_BAL` | Credit limit amount |
| **Payment History** | `PAY_0` to `PAY_6` | Repayment status (Sept-April) |
| **Bill Amounts** | `BILL_AMT1` to `BILL_AMT6` | Monthly bill statements |
| **Payment Amounts** | `PAY_AMT1` to `PAY_AMT6` | Monthly payments |
| **Target** | `default.payment.next.month` | Default in next month (1=Yes, 0=No) |

### Class Distribution
- **No Default (0)**: ~77.88%
- **Default (1)**: ~22.12%
- **Note**: Dataset exhibits class imbalance, addressed using `class_weight='balanced'`

---



## Usage Guide

### Step 1: Launch Notebook
```bash
# Ensure virtual environment is activated
jupyter notebook credit_default.ipynb
# or
code credit_default.ipynb  # if using VS Code
```

### Step 2: Configure MLflow
In your notebook's first cell:
```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_registry_uri("http://127.0.0.1:5000")
mlflow.set_experiment("credit_card_default_lab")
```

### Step 3: Execute Notebook Cells
Run cells sequentially to:
1. **Load and explore data**
2. **Preprocess features**
3. **Train baseline model** (Logistic Regression)
4. **Perform hyperparameter tuning** (Random Forest)
5. **Track experiments** in MLflow
6. **Register best model** to Model Registry
7. **Promote model** to Production stage

### Step 4: Batch Inference
```python
# Load production model
prod_model = mlflow.pyfunc.load_model("models:/credit_default_classifier/Production")

# Score test data
predictions = prod_model.predict(X_test)

# Create scored dataset
scored = X_test.copy()
scored["prob_default"] = predictions
scored.to_csv("scored_test_predictions.csv", index=False)
```

### Step 5: Serve Model (Real-Time Inference)
In a new terminal:
```bash
# Activate virtual environment
source credit_env/bin/activate

# Set MLflow URIs
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_REGISTRY_URI=http://127.0.0.1:5000

# Serve model
mlflow models serve \
  -m models:/credit_default_classifier/Production \
  -h 127.0.0.1 \
  -p 5001
```

### Step 6: REST API Inference
```python
import requests
import numpy as np

url = "http://127.0.0.1:5001/invocations"
payload = {"dataframe_split": X_test.to_dict(orient="split")}

response = requests.post(url, json=payload)
response.raise_for_status()

predictions = np.array(response.json())
print(f"Predictions shape: {predictions.shape}")
print(f"Sample predictions: {predictions[:5]}")
```

---

## Model Performance

### Best Model: Random Forest (Tuned)

| Metric | Validation Set | Test Set |
|--------|---------------|----------|
| **AUC-ROC** | 0.7747 | 0.7794 |
| **Accuracy** | 77.75% | 77.93% |
| **F1 Score** | 0.5347 | 0.5364 |
| **PR-AUC** | 0.5466 | 0.5624 |

### Confusion Matrix (Test Set)
```
              Predicted
              No Default  Default
Actual No      3,910       763
Actual Yes       561       766
```

### Hyperparameters
```python
{
    'n_estimators': 400,
    'max_depth': 10,
    'min_samples_leaf': 3,
    'class_weight': 'balanced'
}
```

---

## API Reference

### REST Endpoint
**URL**: `http://127.0.0.1:5001/invocations`

**Method**: `POST`

**Request Format**:
```json
{
  "dataframe_split": {
    "columns": ["limit_bal", "sex", "education", ...],
    "data": [[20000, 2, 2, 1, 24, ...], ...]
  }
}
```

**Response Format**:
```json
[0.256787, 0.767510, 0.199083, ...]
```

**Example**:
```python
import requests
import pandas as pd

# Prepare data
data = {
    "dataframe_split": X_test.to_dict(orient="split")
}

# Make request
response = requests.post(
    "http://127.0.0.1:5001/invocations",
    json=data,
    headers={"Content-Type": "application/json"}
)

# Get predictions
predictions = response.json()
```

---

## Troubleshooting

### Common Issues

#### 1. MLflow Server Connection Error
```
Error: Connection refused at http://127.0.0.1:5000
```
**Solution**: Ensure MLflow server is running in a separate terminal

#### 2. Module Import Errors
```
ModuleNotFoundError: No module named 'mlflow'
```
**Solution**: Activate virtual environment and reinstall dependencies
```bash
source credit_env/bin/activate
pip install numpy pandas scikit-learn matplotlib seaborn mlflow cloudpickle ipykernel xlrd==1.2.0
```

#### 3. Model Serving Port Conflict
```
Error: Address already in use
```
**Solution**: Use a different port or kill existing process
```bash
# Find process on port 5001
lsof -ti:5001 | xargs kill -9  # macOS/Linux
# or use port 5002
mlflow models serve ... -p 5002
```

#### 4. Dataset Loading Error
```
FileNotFoundError: data/default of credit card clients.xls
```
**Solution**: Verify dataset path is correct (should be in `data/` folder)

#### 5. Prediction Format Issues
If REST API returns predictions in unexpected format, handle both list and dict responses:
```python
data = response.json()
if isinstance(data, dict):
    preds = np.array(data.get("predictions", list(data.values())[0]))
else:
    preds = np.array(data)
```

---

## Notes

- The MLflow server must be running before executing the notebook
- Keep the MLflow server terminal open throughout your session
- Models are stored locally in the `mlruns/` directory
- The SQLite database (`mlflow.db`) tracks all experiments and model versions

---
