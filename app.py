import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import catboost
import xgboost as xgb
import lightgbm as lgb

# Load models (ensure you have saved the models during training)
rf_model = joblib.load("models/random_forest_model.pkl")  # RandomForest model
ab_model = joblib.load("models/adaboost_model.pkl")       # AdaBoost model
cb_model = joblib.load("models/catboost_model.pkl")       # CatBoost model
xgb_model = joblib.load("models/xgboost_model.pkl")       # XGBoost model
lgb_model = joblib.load("models/lgbm_model.pkl")          # LightGBM model

# Load scaler if used
scaler = joblib.load("models/scaler.pkl")  # StandardScaler or any scaler you used

# Streamlit Interface
st.title("Credit Card Fraud Detection System")
st.write("Enter the transaction details below to check for fraud:")

# Collecting user input for transaction features
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
v1 = st.number_input("V1", format="%.6f")
v2 = st.number_input("V2", format="%.6f")
v3 = st.number_input("V3", format="%.6f")
v4 = st.number_input("V4", format="%.6f")
v5 = st.number_input("V5", format="%.6f")
v6 = st.number_input("V6", format="%.6f")
v7 = st.number_input("V7", format="%.6f")
v8 = st.number_input("V8", format="%.6f")
v9 = st.number_input("V9", format="%.6f")
v10 = st.number_input("V10", format="%.6f")
v11 = st.number_input("V11", format="%.6f")
v12 = st.number_input("V12", format="%.6f")
v13 = st.number_input("V13", format="%.6f")
v14 = st.number_input("V14", format="%.6f")
v15 = st.number_input("V15", format="%.6f")
v16 = st.number_input("V16", format="%.6f")
v17 = st.number_input("V17", format="%.6f")
v18 = st.number_input("V18", format="%.6f")
v19 = st.number_input("V19", format="%.6f")
v20 = st.number_input("V20", format="%.6f")
v21 = st.number_input("V21", format="%.6f")
v22 = st.number_input("V22", format="%.6f")
v23 = st.number_input("V23", format="%.6f")
v24 = st.number_input("V24", format="%.6f")
v25 = st.number_input("V25", format="%.6f")
v26 = st.number_input("V26", format="%.6f")
v27 = st.number_input("V27", format="%.6f")
v28 = st.number_input("V28", format="%.6f")
time = st.number_input("Time", format="%.6f")

# Prepare input data as a DataFrame
input_data = pd.DataFrame([[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                            v21, v22, v23, v24, v25, v26, v27, v28, amount, time]],
                          columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                                   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                                   'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Time'])

# Standardize the data (if your model was trained with scaling)
input_data_scaled = scaler.transform(input_data)

# Allow user to select the model
model_choice = st.selectbox("Choose the Model", 
                            ("Random Forest", "AdaBoost", "CatBoost", "XGBoost", "LightGBM"))

# Function to get prediction based on selected model
def get_prediction(model, data):
    prediction = model.predict(data)[0]
    return prediction

# Predict using the selected model
if st.button("Check for Fraud"):
    if model_choice == "Random Forest":
        prediction = get_prediction(rf_model, input_data_scaled)
    elif model_choice == "AdaBoost":
        prediction = get_prediction(ab_model, input_data_scaled)
    elif model_choice == "CatBoost":
        prediction = get_prediction(cb_model, input_data_scaled)
    elif model_choice == "XGBoost":
        prediction = get_prediction(xgb_model, input_data_scaled)
    elif model_choice == "LightGBM":
        prediction = get_prediction(lgb_model, input_data_scaled)
    
    # Display result based on prediction
    if prediction == 1:
        st.subheader("Prediction: Fraudulent Transaction")
        st.error("This transaction is likely to be fraudulent.")
    else:
        st.subheader("Prediction: Legitimate Transaction")
        st.success("This transaction seems legitimate.")
