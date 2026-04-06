import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path


FEATURE_COLUMNS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

class FraudDetector:
    def __init__(self, model_dir='./Model'):
        self.model_dir = Path(model_dir)
        self.load_model_and_scaler()

    def load_model_and_scaler(self):
        """Load the trained model and scaler with error handling"""
        try:
            self.model = joblib.load(self.model_dir / 'anomaly_model.pkl')
            self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
            metadata_path = self.model_dir / 'model_metadata.json'
            self.metadata = {}
            if metadata_path.exists():
                self.metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except Exception as e:
            raise Exception(f"Error loading model or scaler: {str(e)}")

    def validate_data(self, df):
        """Validate input data structure"""
        required_columns = FEATURE_COLUMNS
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
            X = df_processed[FEATURE_COLUMNS].copy()

            # Force numeric conversion for schema safety.
            for column in FEATURE_COLUMNS:
                X[column] = pd.to_numeric(X[column], errors='coerce')

            # Handle missing values
            if X.isnull().any().any():
                X = X.fillna(X.mean())

            if X.isnull().any().any():
                raise ValueError('Input contains non-numeric or invalid values that could not be processed.')
            
            # Scale the features
            X_scaled = self.scaler.transform(X)

            return X_scaled
        except Exception as e:
            raise Exception(f"Error during preprocessing: {str(e)}")

    def make_predictions(self, df):
        """Make predictions on the input data"""
        try:
            X_scaled = self.preprocess_data(df)
            fraud_scores = -self.model.decision_function(X_scaled)

            threshold = self.metadata.get('selected_threshold')
            if threshold is not None:
                predictions = (fraud_scores >= float(threshold)).astype(int)
                prediction_labels = ['Fraudulent' if pred == 1 else 'Not Fraudulent' for pred in predictions]
            else:
                raw_predictions = self.model.predict(X_scaled)
                prediction_labels = [
                    'Fraudulent' if pred == -1 else 'Not Fraudulent' for pred in raw_predictions
                ]

            result_df = df.copy()
            result_df['Anomaly_Score'] = fraud_scores
            result_df['Prediction'] = prediction_labels
            return result_df
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
