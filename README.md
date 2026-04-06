# Credit Card Fraud Detection Project

This project detects fraudulent credit card transactions using a TensorFlow/Keras model and a Streamlit web app.

## Project Files

- [app.py](app.py): Streamlit app for uploading transaction CSV files and generating predictions
- [ml_model.py](ml_model.py): Training script that builds and saves the model and scaler
- [make_csv.py](make_csv.py): Utility script to generate sample transaction input
- [creditcard.csv](creditcard.csv): Training dataset
- [sample_transactions.csv](sample_transactions.csv): Sample input for prediction testing
- [requirements.txt](requirements.txt): Python dependencies
- [architecture.md](architecture.md): High-level architecture documentation

## Prerequisites

- Python 3.9 to 3.12 recommended
- pip

## Setup

1.  Create a virtual environment:

        python -m venv .venv

2.  Activate the virtual environment (Windows PowerShell):

        .\.venv\Scripts\Activate.ps1

3.  Install dependencies:

        pip install -r requirements.txt

## How To Run

### 1. Train the model

Run:

    	python ml_model.py

Expected output artifacts:

- Model/fraud_detection_model.keras
- Model/scaler.pkl

Important note:
The current training script in [ml_model.py](ml_model.py) uses a hardcoded absolute dataset path. If training fails on your machine, update the dataset loading line to use the local [creditcard.csv](creditcard.csv) file in the project root.

### 2. Start the Streamlit app

Run:

    	streamlit run app.py

Then open the URL shown in the terminal, usually:

- http://localhost:8501

### 3. Make predictions

1. Upload a CSV file with required columns:
   - Time
   - V1 through V28
   - Amount
2. Click Predict
3. Review the Prediction results
4. Download the output CSV from the app

## Optional: Generate Sample Input

Run:

    	python make_csv.py

This creates or refreshes [sample_transactions.csv](sample_transactions.csv).

## Input Schema

Required columns:

- Time
- V1 through V28
- Amount

If required columns are missing, the app returns a validation error.

## Troubleshooting

- Error loading model or scaler:
  - Ensure training completed successfully and model files exist in the Model directory
- Streamlit command not found:
  - Confirm your virtual environment is activated before running commands
- TensorFlow installation issues:
  - Verify Python version compatibility and reinstall dependencies from [requirements.txt](requirements.txt)

## Notes

- Prediction threshold is currently fixed in [app.py](app.py)
- Missing values are filled with column means before scaling during inference
