import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from transformers import pipeline
import tensorflow as tf

st.set_page_config(layout="wide")

# Title
st.title("🚀 Predictive Maintenance AI Dashboard")

# File uploader
uploaded_files = st.file_uploader("Upload your sensor datasets (CSV)", type="csv", accept_multiple_files=True)

# Initialize Hugging Face model (example: AI solution generator)
solution_generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=150)

@st.cache_resource
def load_dataset(file):
    df = pd.read_csv(file)
    df = df.ffill().bfill()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.resample('h').mean()
    return df

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@tf.function(reduce_retracing=True)
def predict_lstm(model, X):
    return model(X)

def preprocess_and_predict(df, forecast_steps=30):
    data = df.iloc[:, 0].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    time_steps = 10
    for i in range(time_steps, len(data_scaled) - forecast_steps):
        X.append(data_scaled[i - time_steps:i])
        y.append(data_scaled[i])
    
    X, y = np.array(X), np.array(y)

    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    # Predict future
    last_sequence = data_scaled[-time_steps:]
    future_preds = []
    current_seq = last_sequence.copy()
    
    for _ in range(forecast_steps):
        pred = model.predict(current_seq.reshape(1, time_steps, 1), verbose=0)
        future_preds.append(pred[0][0])
        current_seq = np.append(current_seq[1:], [[pred[0][0]]], axis=0)

    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    return data[-forecast_steps:], scaler.inverse_transform(y), future_preds, model

def generate_ai_diagnosis(df, predictions):
    avg_value = np.mean(predictions)
    prompt = f"The sensor data shows an increasing anomaly trend. Average predicted value is {avg_value:.2f}. What could be the cause and possible maintenance action?"
    response = solution_generator(prompt)[0]['generated_text']
    return response.strip()

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"📊 Dataset: {uploaded_file.name}")
        df = load_dataset(uploaded_file)

        st.write("Preview of data:")
        st.dataframe(df.head())

        with st.spinner("Training LSTM model and making predictions..."):
            real_values, actual_y, predicted_y, model = preprocess_and_predict(df)

        st.subheader("📈 Trend Prediction")
        fig, ax = plt.subplots()
        ax.plot(range(len(actual_y)), actual_y, label='Actual')
        ax.plot(range(len(actual_y), len(actual_y) + len(predicted_y)), predicted_y, label='Forecast')
        ax.legend()
        st.pyplot(fig)

        st.subheader("💡 AI Diagnosis & Solution")
        ai_response = generate_ai_diagnosis(df, predicted_y)
        st.text_area("Suggested Maintenance Action:", value=ai_response, height=200)

else:
    st.info("Please upload one or more CSV sensor datasets to begin.")
