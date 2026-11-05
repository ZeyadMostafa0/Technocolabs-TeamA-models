import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained XGBoost model and features
model = joblib.load('xgb_delinquency_model.pkl')
trained_features = joblib.load('xgb_model_features.pkl')

# App title
st.title("Loan Delinquency Prediction App")

st.markdown("Enter the following loan applicant details from raw data:")

# Input form for raw features
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
orig_upb = st.number_input("Original UPB (Loan Amount)", min_value=0, value=200000)
dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)
ltv = st.number_input("Loan-to-Value Ratio (%)", min_value=0.0, max_value=100.0, value=80.0)
ocltv = st.number_input("Original Combined LTV (%)", min_value=0.0, max_value=100.0, value=85.0)
loan_term = st.number_input("Loan Term (months)", min_value=1, value=360)
mip = st.number_input("Mortgage Insurance Premium (MIP)", min_value=0.0, value=100.0)

# Preprocessing and derived features
monthly_principal = orig_upb / loan_term if loan_term else 0
ltv_ocltv_diff = ltv - ocltv
dti_per_unit = dti / orig_upb if orig_upb else 0
mip_to_loanamt = mip / orig_upb if orig_upb else 0

# Construct final DataFrame for prediction
input_data = {
    'CreditScore': credit_score,
    'OrigUPB': orig_upb,
    'DTI': dti,
    'LTV': ltv,
    'OCLTV': ocltv,
    'LoanTerm': loan_term,
    'MIP': mip,
    'MonthlyPrincipal': monthly_principal,
    'LTV_OCLTV_Diff': ltv_ocltv_diff,
    'DTI_per_Unit': dti_per_unit,
    'MIP_to_LoanAmt': mip_to_loanamt
}

# Create a DataFrame with all expected features
input_df = pd.DataFrame([input_data])

# Ensure all columns match what the model was trained on
for col in trained_features:
    if col not in input_df.columns:
        input_df[col] = 0  # Fill missing engineered/unused columns if any

input_df = input_df[trained_features]  # Ensure correct order

# Predict button
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {'Delinquent' if pred else 'Not Delinquent'}")
    st.write(f"**Probability of Delinquency:** {proba:.2%}")
