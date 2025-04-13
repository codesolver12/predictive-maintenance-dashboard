# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from transformers import pipeline
import io

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
        df.set_index(time_col, inplace=True)

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
    steps_ahead = st.slider("Forecast how many future steps?", min_value=1, max_value=50, value=10)

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

        last_seq = data[-sequence_length:]
        forecast = []

        for _ in range(steps_ahead):
            input_seq = last_seq.reshape((1, sequence_length, 1))
            next_val = lstm_model.predict(input_seq, verbose=0)[0][0]
            forecast.append(next_val)
            last_seq = np.append(last_seq[1:], next_val)

        future_df = pd.DataFrame({f"Predicted {target_col}": forecast})

        st.line_chart(future_df, use_container_width=True)
        st.write("📉 Forecasted Values:")
        st.dataframe(future_df)

        # Download forecast
        csv = future_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", csv, f"forecast_{target_col}.csv", "text/csv")

    # AI Diagnosis & Solutions
    st.subheader("🧠 AI Diagnosis & Solutions")
    description = st.text_input("Describe the issue or anomaly:")
    if st.button("Generate Solution"):
        try:
            ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
            prompt = f"Suggest a realistic predictive maintenance solution for the following issue: {description}"
            response = ai_model(prompt, max_length=100)[0]['generated_text']
            st.success("🩺 Suggested Solution:")
            st.write(response)
        except Exception as e:
            st.warning("⚠️ AI model not supported in current environment. Using rule-based fallback.")
            fallback = {
                "temperature": "Check cooling systems and ventilation.",
                "vibration": "Inspect mechanical joints, consider re-balancing components.",
                "voltage": "Inspect electrical supply and possible short circuits.",
                "default": "Perform general diagnostics and inspect system logs."
            }
            for keyword, advice in fallback.items():
                if keyword.lower() in description.lower():
                    st.success("🩺 Suggested Solution:")
                    st.write(advice)
                    break
            else:
                st.write(fallback["default"])
