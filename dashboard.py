import os
os.environ["STREAMLIT_WATCH_FILE"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, MaxPooling1D, Flatten
from transformers import pipeline

st.set_page_config(page_title="Predictive Maintenance AI Dashboard", layout="wide")
st.title("🚀 AI-Powered Predictive Maintenance Dashboard")

uploaded_file = st.file_uploader("Upload a sensor dataset (.csv)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded!")
    st.dataframe(df.head())

    time_col = st.selectbox("Select timestamp column (if any):", options=["None"] + list(df.columns))
    if time_col != "None":
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(by=time_col).reset_index(drop=True)
    else:
        df["index_time"] = range(len(df))
        time_col = "index_time"

    st.subheader("📊 Dataset Summary")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.write("Detected numerical features:", numerical_cols)

    st.subheader("📈 Visualize Sensor Trends")
    selected_feature = st.selectbox("Choose feature to visualize:", numerical_cols)
    chart_type = st.selectbox("Choose chart type:", ["Line", "Scatter", "Box"])

    fig = go.Figure()
    if chart_type == "Line":
        fig.add_trace(go.Scatter(x=df[time_col], y=df[selected_feature], mode="lines"))
    elif chart_type == "Scatter":
        fig.add_trace(go.Scatter(x=df[time_col], y=df[selected_feature], mode="markers"))
    elif chart_type == "Box":
        fig = go.Figure(data=[go.Box(y=df[selected_feature], name=selected_feature)])

    fig.update_layout(title=f"{chart_type} of {selected_feature}", xaxis_title="Time", yaxis_title=selected_feature)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚠️ Anomaly Detection")
    if len(numerical_cols) > 1:
        model = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = model.fit_predict(df[numerical_cols])
        anomalies = df[df['anomaly'] == -1]
        st.write("Anomalies Detected:", len(anomalies))
        st.dataframe(anomalies)

    st.subheader("🔮 Failure Forecasting with Multiple Models")
    forecast_col = st.selectbox("Select feature for forecasting:", numerical_cols)
    n_steps = st.slider("Steps to forecast:", 1, 20, 5)
    sequence_length = 10
    data = df[forecast_col].ffill().values

    X, y = [], []
    for i in range(len(data) - sequence_length - n_steps):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length:i+sequence_length+n_steps])
    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        st.warning("⚠️ Not enough data.")
    else:
        model_type = st.selectbox("Choose Model Type:", ["LSTM", "GRU", "CNN+LSTM", "Random Forest", "SVR", "XGBoost"])
        if model_type in ["LSTM", "GRU", "CNN+LSTM"]:
            X_dl = X.reshape((X.shape[0], X.shape[1], 1))
            model = Sequential()
            if model_type == "LSTM":
                model.add(LSTM(64, activation='relu', input_shape=(sequence_length, 1)))
            elif model_type == "GRU":
                model.add(GRU(64, activation='relu', input_shape=(sequence_length, 1)))
            elif model_type == "CNN+LSTM":
                model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 1)))
                model.add(MaxPooling1D(pool_size=2))
                model.add(LSTM(64, activation='relu'))
            model.add(Dense(n_steps))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_dl, y, epochs=10, verbose=0)
            input_seq = data[-sequence_length:].reshape(1, sequence_length, 1)
            prediction = model.predict(input_seq)[0]
            y_pred = model.predict(X_dl)
            rmse = np.sqrt(mean_squared_error(y.flatten(), y_pred.flatten()))
        else:
            X_ml = X
            y_ml = y
            if model_type == "Random Forest":
                model = RandomForestRegressor()
            elif model_type == "SVR":
                model = SVR()
            elif model_type == "XGBoost":
                model = XGBRegressor()
            y_flat = y_ml[:, 0]  # predicting first future step for simplicity
            model.fit(X_ml, y_flat)
            prediction = model.predict([data[-sequence_length:]])
            y_pred = model.predict(X_ml)
            rmse = np.sqrt(mean_squared_error(y_flat, y_pred))

        st.success(f"✅ Model Trained: {model_type}")
        st.write(f"📉 RMSE (Root Mean Squared Error): {rmse:.4f}")

        future_index = pd.date_range(start=df[time_col].iloc[-1], periods=n_steps+1, freq='h')[1:] \
            if "datetime" in str(df[time_col].dtype) else list(range(len(df), len(df) + n_steps))

        fig2 = go.Figure()
        fig2.add_scatter(x=df[time_col], y=df[forecast_col], mode='lines', name='Actual')
        fig2.add_scatter(x=future_index, y=prediction[:n_steps], mode='lines', name='Forecast')
        st.plotly_chart(fig2, use_container_width=True)

        forecast_df = pd.DataFrame({time_col: future_index, f"Predicted_{forecast_col}": prediction[:n_steps]})
        st.download_button("📥 Download Forecast", forecast_df.to_csv(index=False), file_name="forecast.csv")

    st.subheader("🧠 AI Diagnosis & Solutions")
    desc = st.text_area("Describe the anomaly or issue:", placeholder="Example: Sudden temperature rise in motor")
    if st.button("Generate AI Solution"):
        try:
            ai_model = pipeline("text-generation", model="gpt2")
            prompt = f"Sensor: {forecast_col}\nAnomaly: {desc}\nSuggest a maintenance solution:"
            response = ai_model(prompt, max_new_tokens=100)[0]['generated_text']
            st.success("🩺 Suggested Solution:")
            st.write(response.strip())
        except Exception:
            st.warning("Fallback AI solution used.")
            fallback = {
                "temperature": "Check for cooling system malfunction or overheating elements.",
                "vibration": "Inspect loose components or imbalance in rotating parts.",
                "voltage": "Check power lines, transformers, or capacitor issues.",
                "pressure": "Inspect valves, seals, and pressure calibration devices.",
                "default": "Run diagnostics and inspect historical sensor logs."
            }
            for key, sol in fallback.items():
                if key in desc.lower():
                    st.success("🩺 Fallback Maintenance Tip:")
                    st.write(sol)
                    break
            else:
                st.write(fallback["default"])
