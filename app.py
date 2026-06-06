import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model(
    "fraud_attention_model.keras",
    compile=False
)

scaler = joblib.load("scaler.pkl")

st.title("Fraud Detection System")

uploaded_file = st.file_uploader(
    "Upload CSV containing 5 transactions",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.write(df.head())

    if len(df) != 5:
        st.error(
            "CSV must contain exactly 5 transactions"
        )

    else:

        scaled = scaler.transform(df)

        sequence = np.expand_dims(
            scaled,
            axis=0
        )

        pred = model.predict(
            sequence,
            verbose=0
        )[0][0]

        st.write(
            f"Fraud Probability: {pred:.4f}"
        )

        if pred > 0.5:
            st.error("Fraud Detected")
        else:
            st.success("Legitimate")