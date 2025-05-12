import streamlit as st
import numpy as np
import joblib

# Load your trained model
model = joblib.load("creditcard.csv")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv("creditcard.csv")

# Prepare the data
X = df.drop("Class", axis=1)
y = df["Class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "fraud_model.pkl")
print("Model saved as fraud_model.pkl")

# Load your trained model
model = joblib.load("fraud_model.pkl")

# Function to make predictions
def predict_fraud(time, v_inputs, amount):
    features = [time] + v_inputs + [amount]
    input_df = pd.DataFrame([features], columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
    return model.predict(input_df)[0]

# Streamlit page setup
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.markdown("<h1 style='text-align: center; color: #003566;'>Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter the transaction details below to detect fraud in real-time.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input Section
st.subheader("Transaction Features")
time = st.number_input("⏱ Time (in seconds since first transaction)", min_value=0.0, format="%.2f")
amount = st.number_input("💰 Transaction Amount ($)", min_value=0.0, format="%.2f")

v_inputs = []
cols = st.columns(4)
for i in range(1, 29):
    with cols[(i - 1) % 4]:
        val = st.number_input(f"V{i}", format="%.6f")
        v_inputs.append(val)

# Prediction
if st.button("🔍 Detect Fraud"):
    features = [time] + v_inputs + [amount]
    input_df = pd.DataFrame([features], columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])

    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("🚨 This is a Fraudulent Transaction!")
    else:
        st.success("✅ This is a Legitimate Transaction.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 13px;'>Developed by Pooja - Final Year Project</div>", unsafe_allow_html=True)

