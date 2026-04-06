# Project Architecture: Credit Card Fraud Detection

## Overview

This project is an anomaly detection system for credit card fraud using Isolation Forest.

The architecture has two primary execution paths:

1. Offline training and evaluation in `ml_model.py`.
2. Online inference in a Streamlit app in `app.py`.

The model is trained only on normal transactions (`Class = 0`) and then used to detect outliers as potential fraud.

## Dataset Schema

Expected dataset columns:

- `Time`
- `V1` through `V28`
- `Amount`
- `Class` (required for training and evaluation)

Label meaning:

- `0` = normal
- `1` = fraud

Inference CSV uploads require feature columns only (`Time`, `V1`-`V28`, `Amount`).

## Main Components

### 1. Data Preparation

- **make_csv.py**
  - Generates sample transaction data for testing inference.
  - Produces `sample_transactions.csv` with required feature columns.

### 2. Model Training and Evaluation

- **ml_model.py**
  - Loads and validates `creditcard.csv`.
  - Normalizes schema and coerces required columns to numeric.
  - Uses a stratified validation split for contamination tuning.
  - Tunes contamination over candidate values using PR-AUC.
  - Fits `StandardScaler` on normal-only rows (`Class == 0`).
  - Trains `IsolationForest` (`n_estimators=100`, selected contamination, `random_state=42`).
  - Computes anomaly scores (`-decision_function`).
  - Selects a score threshold from the precision-recall curve (best fraud F1).
  - Evaluates with:
    - confusion matrix
    - classification report
    - PR-AUC
    - ROC-AUC
  - Saves artifacts into `./Model/`.

### 3. Prediction and Inference

- **app.py**
  - Streamlit web interface for CSV upload and batch prediction.
  - Loads saved model and scaler from `./Model/`.
  - Optionally loads `model_metadata.json` for tuned threshold use.
  - Validates required feature columns.
  - Applies numeric coercion and scaling.
  - Predicts fraud via:
    - tuned threshold on anomaly scores when metadata exists
    - fallback to model default labels otherwise
  - Outputs:
    - `Anomaly_Score`
    - `Prediction` (`Fraudulent` or `Not Fraudulent`)
  - Supports CSV result download.

### 4. Entry Point

- **main.py**
  - Placeholder script (not part of model training/inference flow).

### 5. Configuration and Dependencies

- **requirements.txt**
  - Python dependencies for pandas, scikit-learn, Streamlit, joblib, and related packages.
- **pyproject.toml**
  - Project metadata and packaging configuration.

### 6. Model Artifacts

Generated in `./Model/`:

- `anomaly_model.pkl` (trained Isolation Forest)
- `scaler.pkl` (fitted StandardScaler)
- `model_metadata.json` (selected contamination, threshold, metrics, schema metadata)

## Data Flow

1. **Data Preparation**
   - `make_csv.py` -> `sample_transactions.csv`
2. **Training and Tuning**
   - `ml_model.py` -> load/validate dataset -> tune contamination -> train model on normal rows -> evaluate -> save artifacts
3. **Inference**
   - User uploads CSV in Streamlit (`app.py`) -> validate and preprocess -> score and classify -> display/download predictions

## Technology Stack

- Python 3.13+
- scikit-learn (`IsolationForest`, metrics, split utilities)
- pandas
- NumPy
- Streamlit
- joblib

## Run Sequence

1. Train model and create artifacts:
   - `python ml_model.py`
   - or `uv run ml_model.py`
2. Start inference app:
   - `streamlit run app.py`
3. Upload CSV with required feature columns and review predictions.
