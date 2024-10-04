
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# Replace 'your_data_file.csv' with the path to your dataset file
data = pd.read_csv('your_data_file.csv')

# Example data preprocessing
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Reshape data for LSTM [samples, time steps, features]
    reshaped_data = np.reshape(scaled_data, (scaled_data.shape[0], 1, scaled_data.shape[1]))
    return reshaped_data, scaler



def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Assuming 'data' is your loaded dataset
preprocessed_data, scaler = preprocess_data(data)
train_data, test_data = train_test_split(preprocessed_data, test_size=0.2)

model = build_lstm_model(input_shape=(1, train_data.shape[2]))
model.fit(train_data, epochs=100, batch_size=32)

