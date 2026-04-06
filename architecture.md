# Credit Card Fraud Detection Project Architecture

## 1. Overview

This project detects fraudulent credit card transactions using:

- A TensorFlow/Keras CNN model for binary classification
- A Streamlit app for CSV upload and prediction
- A saved scaler to keep inference preprocessing aligned with training

Main source files:

- app.py
- ml_model.py
- make_csv.py
- requirements.txt

## 2. Components

### 2.1 Model Training

Implemented in `ml_model.py`

Responsibilities:

- Load dataset
- Split features and target
- Train/test split
- Standardize features with StandardScaler
- Train a 1D CNN model
- Save artifacts into `Model` directory:
  - `fraud_detection_model.keras`
  - `scaler.pkl`

### 2.2 Inference and UI

Implemented in `app.py`

Responsibilities:

- Streamlit interface for CSV upload
- Input validation for required columns:
  - `Time`, `V1..V28`, `Amount`
- Missing value handling using mean imputation
- Feature scaling using saved scaler
- Reshaping for Conv1D input
- Prediction and labeling:
  - score >= 0.1 -> Fraudulent
  - otherwise -> Not Fraudulent
- Display and CSV download of results

### 2.3 Sample Data Generation

Implemented in `make_csv.py`

Responsibilities:

- Creates sample input rows with required schema
- Saves `sample_transactions.csv`

## 3. Data Flow

### 3.1 Training Flow

1. Load transactions from `creditcard.csv`
2. Separate `X` and `y`, where `y` is `Class`
3. Perform train-test split
4. Fit scaler on training features
5. Transform train and test features
6. Reshape to 3D tensor for Conv1D
7. Train CNN model
8. Save model and scaler artifacts

### 3.2 Inference Flow

1. User uploads CSV in Streamlit
2. Validate required columns
3. Fill missing values
4. Scale using saved scaler
5. Reshape for Conv1D
6. Predict fraud probabilities
7. Apply threshold and append `Prediction` column
8. Show results and allow CSV download

## 4. Model Definition

Defined in `ml_model.py`:

- Conv1D(64) + BatchNorm + MaxPool + Dropout
- Conv1D(128) + BatchNorm + MaxPool + Dropout
- Flatten + Dense(128) + Dropout
- Dense(1, sigmoid)

Training setup:

- Loss: binary_crossentropy
- Optimizer: Adam (0.001)
- Metric: accuracy
- Epochs: 5

## 5. Dependencies

Dependencies are pinned in `requirements.txt`, including:

- tensorflow
- keras
- scikit-learn
- pandas
- numpy
- streamlit
- joblib

## 6. Artifacts

Artifacts produced during training:

- `Model/fraud_detection_model.keras`
- `Model/scaler.pkl`

Artifacts used for inference:

- Uploaded transaction CSV from user
- Prediction output CSV generated in app

## 7. Operational Notes

- Model artifacts must exist in `Model` before running inference
- Input CSV must contain all required feature columns
- Inference preprocessing must match training scaler

## 8. Current Limitations

- `ml_model.py` currently uses a hardcoded absolute path for dataset loading
- Threshold is fixed in code
- No automated tests currently included

## 9. Suggested Improvements

1. Replace hardcoded dataset path with a relative/CLI-configured path
2. Add evaluation metrics (precision, recall, F1, ROC-AUC)
3. Make prediction threshold configurable in Streamlit UI
4. Add unit tests for validation and preprocessing
5. Add model/version metadata to saved artifacts
