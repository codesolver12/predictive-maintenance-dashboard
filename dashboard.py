# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

    # Auto Detection
    st.subheader("📊 Dataset Summary & Feature Detection")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.write("Detected numerical features:", numerical_cols)

    # Plot Trends
    if numerical_cols:
        st.subheader("📈 Visualize Sensor Trends")
        selected = st.selectbox("Choose feature to visualize:", numerical_cols)
        fig = px.line(df, y=selected, title=f"{selected} over Time")
        st.plotly_chart(fig)

    # Anomaly Detection
    st.subheader("⚠️ Anomaly Detection")
    if len(numerical_cols) > 1:
        model = IsolationForest(contamination=0.05)
        df['anomaly'] = model.fit_predict(df[numerical_cols])
        st.write("Anomalies (marked as -1):")
        st.dataframe(df[df['anomaly'] == -1])
        st.write("📌 Total Anomalies Detected:", sum(df['anomaly'] == -1))

    # LSTM Failure Prediction
    st.subheader("🔮 Predict Future Failures")
    target_col = st.selectbox("Select feature to predict future trends:", numerical_cols)
    if target_col:
        sequence_length = 10
        data = df[target_col].fillna(method='ffill').values
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        lstm_model = Sequential([
            LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(X, y, epochs=10, verbose=0)

        future = lstm_model.predict(X[-10:].reshape(1, sequence_length, 1))
        st.write(f"🧭 Predicted future {target_col}: {future[0][0]:.2f}")

    # AI Diagnosis & Solutions
    st.subheader("🧠 AI Diagnosis & Solutions")
    description = st.text_input("Describe the issue or anomaly (e.g., high vibration, low voltage, etc.):")

    if st.button("Generate Solution"):
        try:
            # Refined expert-style prompt
            prompt = f"""
            You are an expert in predictive maintenance and diagnostics for industrial machines.
            Based on the following sensor anomaly description, provide a clear, actionable maintenance solution.

            Issue Description: {description}

            Include the most likely root cause and what a maintenance engineer should do.
            """

            ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
            response = ai_model(prompt, max_length=150)[0]['generated_text']

            if "not sure" in response or len(response.strip()) < 20:
                raise ValueError("Generic or unclear response.")

            st.success("🩺 Suggested Solution:")
            st.write(response)

        except Exception as e:
            st.warning("⚠️ AI model not available or gave a generic result. Using rule-based solution.")
            fallback = {
                "temperature": "Overheating detected. Check for clogged vents, broken fans, or high ambient temperatures.",
                "vibration": "Abnormal vibrations. Inspect bearings, alignment, and unbalanced rotating parts.",
                "voltage": "Voltage fluctuation. Examine power input, wiring integrity, and circuit protection.",
                "pressure": "Pressure anomaly. Inspect valves, pumps, or leakages in the system.",
                "current": "Current spike detected. Possible motor overload or short circuit.",
                "speed": "Speed inconsistency. Check motor drives, belts, or control signals.",
                "default": "Perform full system diagnostic. Inspect logs, signals, and historical trends for root cause."
            }

            for keyword, advice in fallback.items():
                if keyword.lower() in description.lower():
                    st.success("🩺 Suggested Solution:")
                    st.write(advice)
                    break
            else:
                st.write(fallback["default"])
