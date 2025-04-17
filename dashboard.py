# 🔁 [NO CHANGES] Imports & Configs
import os
os.environ["STREAMLIT_WATCH_FILE"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from transformers import pipeline

st.set_page_config(page_title="Predictive Maintenance AI Dashboard", layout="wide")
st.title("🚀 AI-Powered Predictive Maintenance Dashboard")

# 🔁 Upload Section
uploaded_file = st.file_uploader("Upload a sensor dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded!")
    st.write("Preview:")
    st.dataframe(df.head())

    # 🔁 Timestamp Handling
    time_col = st.selectbox("Select timestamp column (if any):", options=["None"] + list(df.columns))
    if time_col != "None":
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(by=time_col).reset_index(drop=True)
    else:
        df["index_time"] = range(len(df))
        time_col = "index_time"

    # 🔁 Feature Detection
    st.subheader("📊 Dataset Summary & Feature Detection")
    st.write("Shape:", df.shape)
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.write("Detected numerical features:", numerical_cols)

    # 🔁 Sensor Visualization
    st.subheader("📈 Visualize Sensor Trends")
    selected_feature = st.selectbox("Choose a feature to visualize:", numerical_cols)
    chart_type = st.selectbox("Choose chart type:", ["Line", "Scatter", "Bar", "Box", "Histogram"])

    fig = go.Figure()
    if chart_type == "Line":
        fig.add_trace(go.Scatter(x=df[time_col], y=df[selected_feature], mode="lines", name="Line"))
    elif chart_type == "Scatter":
        fig.add_trace(go.Scatter(x=df[time_col], y=df[selected_feature], mode="markers", name="Scatter"))
    elif chart_type == "Bar":
        fig.add_trace(go.Bar(x=df[time_col], y=df[selected_feature], name="Bar"))
    elif chart_type == "Box":
        fig = go.Figure(data=[go.Box(y=df[selected_feature], name=selected_feature)])
    elif chart_type == "Histogram":
        fig = go.Figure(data=[go.Histogram(x=df[selected_feature], name=selected_feature)])
    fig.update_layout(title=f"{chart_type} Plot of {selected_feature}", xaxis_title="Time", yaxis_title=selected_feature)
    st.plotly_chart(fig, use_container_width=True)

    # 🔁 Anomaly Detection
    st.subheader("⚠️ Anomaly Detection using Isolation Forest")
    if len(numerical_cols) > 1:
        model = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = model.fit_predict(df[numerical_cols])
        anomaly_points = df[df['anomaly'] == -1]
        st.write("Detected anomalies (-1):")
        st.dataframe(anomaly_points)
        st.write("📌 Total Anomalies Detected:", len(anomaly_points))
        if chart_type in ["Line", "Scatter", "Bar"]:
            fig.add_trace(go.Scatter(
                x=anomaly_points[time_col],
                y=anomaly_points[selected_feature],
                mode="markers", name="Anomalies", marker=dict(color="red", size=6)))
            st.plotly_chart(fig, use_container_width=True)
        st.download_button("📤 Download Anomalies CSV", anomaly_points.to_csv(index=False),
                           file_name="anomalies.csv", mime="text/csv")

    # 🔁 Forecasting with Model Choice
    st.subheader("🔮 Multi-step Failure Forecasting")

    forecast_col = st.selectbox("Select feature for prediction:", numerical_cols)
    n_steps = st.slider("Select number of future steps to forecast:", 1, 20, 5)
    model_type = st.radio("Choose model type:", ["Deep Learning (LSTM)", "Machine Learning (Linear Regression)"])

    data = df[forecast_col].ffill().values
    sequence_length = 10

    X, y = [], []
    for i in range(len(data) - sequence_length - n_steps):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length:i + sequence_length + n_steps])
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        st.warning("⚠️ Not enough data for prediction. Please upload a longer dataset.")
    else:
        if model_type == "Deep Learning (LSTM)":
            X_dl = X.reshape((X.shape[0], X.shape[1], 1))
            model = Sequential([
                LSTM(64, activation='relu', input_shape=(sequence_length, 1)),
                Dense(n_steps)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_dl, y, epochs=10, verbose=0)
            input_seq = data[-sequence_length:].reshape(1, sequence_length, 1)
            predictions = model.predict(input_seq)[0]
            y_pred_sample = model.predict(X_dl)
            mse = mean_squared_error(y.flatten(), y_pred_sample.flatten())
        else:
            # Flatten for ML (Linear Regression)
            X_ml = X
            y_ml = y
            model = LinearRegression()
            model.fit(X_ml, y_ml)
            input_seq = data[-sequence_length:].reshape(1, -1)
            predictions = model.predict(input_seq)[0]
            y_pred_sample = model.predict(X_ml)
            mse = mean_squared_error(y.flatten(), y_pred_sample.flatten())

        # Plot predictions
        future_index = pd.date_range(start=df[time_col].iloc[-1], periods=n_steps + 1, freq='h')[1:] \
            if "datetime" in str(df[time_col].dtype) else list(range(len(df), len(df) + n_steps))
        fig2 = go.Figure()
        fig2.add_scatter(x=df[time_col], y=df[forecast_col], mode='lines', name='Actual')
        fig2.add_scatter(x=future_index, y=predictions, mode='lines', name='Predicted')
        fig2.update_layout(title="📉 Trend Forecasting", xaxis_title="Time", yaxis_title=forecast_col)
        st.plotly_chart(fig2, use_container_width=True)

        forecast_df = pd.DataFrame({time_col: future_index, f"Predicted_{forecast_col}": predictions})
        st.subheader("📊 Forecast Results & Accuracy")
        st.dataframe(forecast_df)
        st.write(f"✅ **Model Accuracy (MSE):** {mse:.4f}")

        st.download_button("📥 Download Forecast CSV", data=forecast_df.to_csv(index=False),
                           file_name="forecast_results.csv", mime="text/csv")

    # 🔁 AI Diagnosis
    st.subheader("🧠 AI Diagnosis & Solutions")
    description = st.text_area("Describe the issue or anomaly:", placeholder="Example: sudden temperature rise and frequent vibration")
    if st.button("Generate AI Solution"):
        try:
            ai_model = pipeline("text-generation", model="gpt2")
            prompt = f"Sensor Feature: {forecast_col}\nAnomaly Description: {description}\nSuggest a detailed predictive maintenance solution:"
            response = ai_model(prompt, max_new_tokens=100)[0]['generated_text']
            st.success("🩺 Suggested AI-Based Solution:")
            st.write(response.strip())
        except Exception:
            st.warning("⚠️ AI model not supported or failed. Using fallback solution.")
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
