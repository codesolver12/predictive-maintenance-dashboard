# dashboard.py

import os
os.environ["STREAMLIT_WATCH_FILE"] = "false"  # Disable file watcher to avoid torch errors

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from transformers import pipeline

st.set_page_config(page_title="Predictive Maintenance AI Dashboard", layout="wide")
st.title("🚀 AI-Powered Predictive Maintenance Dashboard")

# Upload Section
uploaded_file = st.file_uploader("Upload a sensor dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded!")
    st.write("Preview:")
    st.dataframe(df.head())

    # Handle Timestamps
    time_col = st.selectbox("Select timestamp column (if any):", options=["None"] + list(df.columns))
    if time_col != "None":
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(by=time_col)
        df = df.reset_index(drop=True)
    else:
        df["index_time"] = range(len(df))
        time_col = "index_time"

    # Feature Detection
    st.subheader("📊 Dataset Summary & Feature Detection")
    st.write("Shape:", df.shape)
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.write("Detected numerical features:", numerical_cols)

    # Visualize Trends
    st.subheader("📈 Visualize Sensor Trends")
    selected_feature = st.selectbox("Choose a feature to visualize:", numerical_cols)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[time_col], y=df[selected_feature], mode="lines", name="Sensor Data"))
    fig.update_layout(title=f"{selected_feature} Over Time", xaxis_title="Time", yaxis_title=selected_feature)
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly Detection
    st.subheader("⚠️ Anomaly Detection using Isolation Forest")
    if len(numerical_cols) > 1:
        model = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = model.fit_predict(df[numerical_cols])
        st.write("Detected anomalies (-1):")
        st.dataframe(df[df['anomaly'] == -1])
        st.write("📌 Total Anomalies Detected:", sum(df['anomaly'] == -1))

    # Predictive Modeling
    st.subheader("🔮 Multi-step LSTM Failure Forecasting")
    forecast_col = st.selectbox("Select feature for prediction:", numerical_cols)
    n_steps = st.slider("Select number of future steps to forecast:", 1, 20, 5)

    data = df[forecast_col].fillna(method='ffill').values
    sequence_length = 10

    X, y = [], []
    for i in range(len(data) - sequence_length - n_steps):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length:i + sequence_length + n_steps])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    if len(X) > 0:
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(sequence_length, 1)),
            Dense(n_steps)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, verbose=0)

        input_seq = data[-sequence_length:].reshape(1, sequence_length, 1)
        predictions = model.predict(input_seq)[0]

        future_index = pd.date_range(start=df[time_col].iloc[-1], periods=n_steps+1, freq='H')[1:] \
                       if "datetime" in str(df[time_col].dtype) else list(range(len(df), len(df) + n_steps))

        fig2 = go.Figure()
        fig2.add_scatter(x=df[time_col], y=df[forecast_col], mode='lines', name='Actual')
        fig2.add_scatter(x=future_index, y=predictions, mode='lines', name='Predicted')
        fig2.update_layout(title="📉 Trend Forecasting", xaxis_title="Time", yaxis_title=forecast_col)
        st.plotly_chart(fig2, use_container_width=True)

        forecast_df = pd.DataFrame({time_col: future_index, f"Predicted_{forecast_col}": predictions})
        st.subheader("📤 Export Predicted Results")
        st.dataframe(forecast_df)
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast_results.csv", mime="text/csv")

    # AI Diagnosis
    st.subheader("🧠 AI Diagnosis & Solutions")
    description = st.text_area("Describe the issue or anomaly:", placeholder="Example: sudden temperature rise and frequent vibration")
    if st.button("Generate AI Solution"):
        try:
            ai_model = pipeline("text-generation", model="gpt2")
            prompt = f"Suggest a predictive maintenance solution for the following anomaly: {description}"
            response = ai_model(prompt, max_new_tokens=100)[0]['generated_text']
            st.success("🩺 Suggested AI-Based Solution:")
            st.write(response.strip())
        except Exception as e:
            st.warning("⚠️ AI model not supported or failed. Using fallback.")
            fallback = {
                "temperature": "Check cooling systems, thermal paste, and ambient environment.",
                "vibration": "Inspect rotating parts and mounting bolts. Consider rebalancing.",
                "voltage": "Check power supply units and circuit integrity.",
                "pressure": "Inspect valves and sensor calibration.",
                "default": "Run general diagnostics and inspect historical logs for anomalies."
            }
            for key, val in fallback.items():
                if key in description.lower():
                    st.success("🩺 Suggested Fallback Solution:")
                    st.write(val)
                    break
            else:
                st.write(fallback["default"])
