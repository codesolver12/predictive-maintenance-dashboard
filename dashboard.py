import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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

        if len(data) > sequence_length + 10:
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

            last_sequence = data[-sequence_length:].reshape(1, sequence_length, 1)
            future = lstm_model.predict(last_sequence)
            st.write(f"🧭 Predicted future {target_col}: {future[0][0]:.2f}")
        else:
            st.warning("📉 Not enough data points for future prediction. Please upload a longer dataset.")

    # Diagnosis with fallback AI or rule-based suggestion
    st.subheader("🧠 AI Diagnosis & Solutions")
    description = st.text_input("Describe the issue or anomaly:")
    if st.button("Generate Solution"):
        context = f"Sensor features: {', '.join(numerical_cols)}. Anomaly Description: {description}"
        try:
            from transformers import pipeline
            ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
            prompt = f"Suggest a maintenance solution for the following issue in a sensor system:\n{context}"
            response = ai_model(prompt, max_length=100)[0]['generated_text']
            st.success("🩺 Suggested Solution:")
            st.write(response)
        except Exception as e:
            st.warning("⚠️ AI model not supported in current environment. Using rule-based fallback.")
            fallback = {
                "temperature": "Check cooling systems and ventilation.",
                "vibration": "Inspect mechanical joints, consider re-balancing components.",
                "voltage": "Inspect electrical supply and possible short circuits.",
                "pressure": "Check for leaks or faulty pressure valves.",
                "current": "Check current sensors and system load balance.",
                "default": "Perform general diagnostics and inspect system logs."
            }
            found = False
            for keyword, advice in fallback.items():
                if keyword.lower() in description.lower():
                    st.success("🩺 Suggested Solution:")
                    st.write(advice)
                    found = True
                    break
            if not found:
                st.write(fallback["default"])

