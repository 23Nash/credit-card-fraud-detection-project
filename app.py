import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from pathlib import Path

class FraudDetector:
    def __init__(self, model_dir='./Model'):
        self.model_dir = Path(model_dir)
        self.load_model_and_scaler()

    def load_model_and_scaler(self):
        """Load the trained model and scaler with error handling"""
        try:
            self.model = load_model(self.model_dir / 'fraud_detection_model.keras')
            self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
        except Exception as e:
            raise Exception(f"Error loading model or scaler: {str(e)}")

    def validate_data(self, df):
        """Validate input data structure"""
        required_columns = ['Time'] + [f'V{i}' for i in range(1, 29)]+[ 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return True

    def preprocess_data(self, df):
        """Preprocess the input data"""
        try:
            # Create a copy to avoid modifying the original dataframe
            df_processed = df.copy()
            
            # Select only the required features
            feature_columns = ['Time'] + [f'V{i}' for i in range(1, 29)]+['Amount']
            X = df_processed[feature_columns]
            
            # Handle missing values
            if X.isnull().any().any():
                X = X.fillna(X.mean())
            
            # Scale the features
            X_scaled = self.scaler.transform(X)
            
            # Reshape for CNN
            X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
            
            return X_scaled
        except Exception as e:
            raise Exception(f"Error during preprocessing: {str(e)}")

    def make_predictions(self, df):
        """Make predictions on the input data"""
        try:
            X_scaled = self.preprocess_data(df)
            predictions = self.model.predict(X_scaled)
            df['Prediction'] = ['Fraudulent' if pred >= 0.1 else 'Not Fraudulent' for pred in predictions]
            return df
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")

# Streamlit app
st.title('Credit Card Fraud Detection')

st.write("""
## Upload a CSV file to check for fraudulent transactions
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Input Data")
    st.write(df.head())

    detector = FraudDetector()

    if st.button('Predict'):
        try:
            detector.validate_data(df)
            result_df = detector.make_predictions(df)
            st.write("### Prediction Results")
            st.write(result_df)
            st.download_button(
                label="Download Predictions",
                data=result_df.to_csv(index=False).encode('utf-8'),
                file_name='predictions.csv',
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
