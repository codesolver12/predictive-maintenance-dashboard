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
        df = df.set_index(time_col)

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

    # LSTM Failure Prediction with Multiple-step Forecast and Plot
    st.subheader("🔮 Predict Future Failures")
    target_col = st.selectbox("Select feature to predict future trends:", numerical_cols)
    if target_col:
        sequence_length = 10
        forecast_steps = st.slider("Select number of future steps to predict:", 1, 20, 5)

        data = df[target_col].fillna(method='ffill').values
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        X, y = np.array(X), np.array(y)
        if len(X) > 0:
            X = X.reshape((X.shape[0], X.shape[1], 1))

            lstm_model = Sequential([
                LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X, y, epochs=10, verbose=0)

            input_seq = data[-sequence_length:]
            predictions = []
            for _ in range(forecast_steps):
                seq_input = input_seq.reshape((1, sequence_length, 1))
                pred = lstm_model.predict(seq_input, verbose=0)[0][0]
                predictions.append(pred)
                input_seq = np.append(input_seq[1:], pred)

            st.write(f"🧭 Predicted future {target_col} values:")
            st.write(predictions)

            # Plot actual + predicted values
            past = data[-50:]
            all_data = np.concatenate([past, predictions])
            past_index = list(range(len(past)))
            future_index = list(range(len(past), len(past) + forecast_steps))

            import plotly.graph_objects as go
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=past, x=past_index, mode='lines', name='Actual'))
            fig2.add_trace(go.Scatter(y=predictions, x=future_index, mode='lines', name='Predicted'))
            fig2.update_layout(title="📉 Trend Forecasting", xaxis_title="Index", yaxis_title=target_col)
            st.plotly_chart(fig2)

            # Save predictions
            if st.button("💾 Export Predictions as CSV"):
                output_df = pd.DataFrame({
                    'Step': list(range(1, forecast_steps + 1)),
                    f'Predicted_{target_col}': predictions
                })
                csv = output_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f'predicted_{target_col}.csv',
                    mime='text/csv',
                )

    # Diagnosis with fallback AI or rule-based suggestion
    st.subheader("🧠 AI Diagnosis & Solutions")
    description = st.text_input("Describe the issue or anomaly:")
    if st.button("Generate Solution"):
        try:
            ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
            prompt = f"Suggest a realistic and technical predictive maintenance solution for: {description}"
            response = ai_model(prompt, max_length=100)[0]['generated_text']
            st.success("🩺 Suggested Solution:")
            st.write(response)
        except Exception:
            # Simple fallback if transformer fails
            fallback = {
                "temperature": "Check cooling systems and ensure proper ventilation around sensors.",
                "vibration": "Inspect bearings, motors, and mounting parts for looseness or wear.",
                "voltage": "Examine power supply stability, grounding, and wiring connections.",
                "default": "Perform general diagnostics, verify sensor calibration, and inspect recent logs."
            }
            for keyword, advice in fallback.items():
                if keyword.lower() in description.lower():
                    st.success("🩺 Suggested Solution:")
                    st.write(advice)
                    break
            else:
                st.write(fallback["default"])
