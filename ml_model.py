import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv1D, MaxPool1D, Conv2D
from sklearn.preprocessing import StandardScaler
import os

# Load the dataset
data = pd.read_csv(r'C:\Users\Sanyu Tuscano\Desktop\Project\Project\creditcard.csv')

# Analyze the distribution of the 'Class' column
class_distribution = data["Class"].value_counts()
print("Class distribution in the dataset:")
print(class_distribution)

# Preprocess the data
X = data.drop(columns=['Class'])
y = data['Class']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to DataFrame and then to NumPy arrays
X_train = pd.DataFrame(X_train).to_numpy()
X_test = pd.DataFrame(X_test).to_numpy()

# Reshape the data to match the expected input shape of the model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the ML model
model = Sequential()
model.add(Conv1D(64, 2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.2))

model.add(Conv1D(128, 2, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                   epochs=5,  # adjust as needed
                   batch_size=32,
                   validation_split=0.2,
                   verbose=1)
# Ensure the Model directory exists
model_dir = './Model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the model using TensorFlow's native method
model_path = os.path.join(model_dir, 'fraud_detection_model.keras')
model.save(model_path)

# Save only the scaler with joblib
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))