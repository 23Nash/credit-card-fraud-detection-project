# Credit Card Fraud Detection (Isolation Forest)

This project detects potentially fraudulent credit card transactions using an anomaly detection approach based on Isolation Forest.

The system is designed around a simple two-step workflow:

1. Train an anomaly model on normal transactions only.
2. Run a Streamlit app to score uploaded transaction files.

## Project Overview

The dataset follows the common credit card fraud format:

- Features: Time, V1-V28, Amount
- Target: Class
  - 0 = normal transaction
  - 1 = fraud transaction

Model strategy:

- Training uses only rows where Class == 0.
- Isolation Forest learns normal behavior and flags outliers as anomalies.
- During prediction:
  - -1 = Fraudulent
  - 1 = Not Fraudulent

## Repository Structure

- app.py: Streamlit interface for CSV upload and fraud prediction.
- ml_model.py: Training, evaluation, and model artifact export.
- make_csv.py: Generates a sample CSV for testing.
- creditcard.csv: Training/evaluation dataset.
- sample_transactions.csv: Example input file for inference.
- Model/: Created automatically; stores trained artifacts.

## How the Pipeline Works

### Training (ml_model.py)

1. Loads creditcard.csv.
2. Validates required columns:
   - Time, V1-V28, Amount, Class
3. Cleans and normalizes schema:
   - trims column-name formatting
   - enforces numeric values
   - drops invalid rows
4. Creates a stratified validation split for tuning.
5. Tunes contamination over candidate values using PR-AUC.
6. Fits StandardScaler on only normal rows (Class == 0).
7. Trains final IsolationForest with:
   - n_estimators = 100
   - selected contamination from tuning
   - random_state = 42
8. Computes anomaly scores and selects an F1-optimized score threshold.
9. Evaluates on full dataset using:
   - confusion matrix
   - classification report
   - PR-AUC and ROC-AUC
10. Saves artifacts:

- Model/anomaly_model.pkl
- Model/scaler.pkl
- Model/model_metadata.json

### Inference (app.py)

1. Loads Model/anomaly_model.pkl and Model/scaler.pkl.
2. Validates uploaded CSV columns:
   - Time, V1-V28, Amount
3. Applies same scaling with the saved scaler.
4. Computes anomaly scores and applies the tuned threshold if metadata exists.
5. Displays anomaly score and prediction, and allows download of predictions.

## Setup and Run

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the anomaly detection model

```bash
python ml_model.py
```

or:

```bash
uv run ml_model.py
```

Expected output:

- evaluation logs (confusion matrix and classification report)
- generated model files in Model/
- metadata with selected threshold in Model/model_metadata.json

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, upload a CSV file, and click Predict.

## Input CSV Format for the App

Required columns for inference:

- Time
- V1 through V28
- Amount

Notes:

- Class is optional for inference uploads.
- Extra columns are ignored by the model preprocessing logic.
- Output now includes Anomaly_Score and Prediction.

## Generate a Sample Inference File

```bash
python make_csv.py
```

This generates sample_transactions.csv, which can be uploaded to the app.

## Configuration

In ml_model.py, contamination is configurable through:

- CONTAMINATION = 0.002

You can tune this value based on expected fraud prevalence and precision/recall trade-offs.

## Troubleshooting

- Model files missing:
  - Run python ml_model.py first to create Model/anomaly_model.pkl and Model/scaler.pkl.
- Import errors in editor:
  - Ensure your active Python interpreter is the same one where requirements were installed.
- CSV validation errors:
  - Verify exact required feature names and types.

## Tech Stack

- Python
- pandas
- scikit-learn
- Streamlit
- joblib
