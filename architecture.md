# Project Architecture: Credit Card Fraud Detection

## Overview

This project is a machine learning-based credit card fraud detection system. It includes data preprocessing, model training, and a web-based interface for users to upload transaction data and receive fraud predictions.

## Main Components

### 1. Data Preparation

- **make_csv.py**: Generates or manipulates CSV files for sample transactions. It creates a DataFrame with the required features for the model.
- **sample_transactions.csv**: Example CSV file with transaction data, used for testing or demonstration.

### 2. Model Training

- **ml_model.py**: Handles the end-to-end process of loading the dataset, preprocessing (scaling, reshaping), defining and training a 1D CNN model, and saving both the trained model and scaler to the `./Model` directory.
  - Uses Keras, TensorFlow, and scikit-learn.
  - Saves model as `fraud_detection_model.keras` and scaler as `scaler.pkl`.

### 3. Prediction & Inference

- **app.py**: Implements a Streamlit web app for user interaction.
  - Loads the trained model and scaler.
  - Validates and preprocesses uploaded CSV data.
  - Makes predictions and displays/downloads results.
  - Handles errors gracefully and provides user feedback.
  - Core logic is encapsulated in the `FraudDetector` class.

### 4. Entry Point

- **main.py**: Simple script that prints a greeting. (Placeholder, not used in main workflow.)

### 5. Configuration & Dependencies

- **requirements.txt**: Lists all Python dependencies (TensorFlow, scikit-learn, pandas, Streamlit, etc.).
- **pyproject.toml**: Basic project metadata.

### 6. Model Artifacts

- **./Model/**: Directory (created at runtime) where the trained model and scaler are stored.

## Data Flow Diagram

1. **Data Preparation**
   - `make_csv.py` → `sample_transactions.csv`
2. **Model Training**
   - `ml_model.py` → Reads raw data → Preprocesses → Trains model → Saves model/scaler
3. **Prediction (Web App)**
   - User uploads CSV via Streamlit (`app.py`)
   - Data validated/preprocessed → Model predicts → Results shown/downloadable

## Technologies Used

- Python 3.13+
- TensorFlow/Keras
- scikit-learn
- pandas
- Streamlit
- joblib

## File Structure

- `app.py` — Streamlit web app for predictions
- `ml_model.py` — Model training and saving
- `make_csv.py` — Sample CSV generation
- `main.py` — Placeholder entry point
- `requirements.txt` — Python dependencies
- `pyproject.toml` — Project metadata
- `sample_transactions.csv` — Example data
- `./Model/` — Saved model and scaler

## Usage

1. Train the model with `ml_model.py` (ensure dataset path is correct).
2. Run the web app with `streamlit run app.py`.
3. Upload a CSV file with the required columns to get predictions.
