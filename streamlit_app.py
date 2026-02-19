import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Loan Default Prediction", layout="centered")

st.title("🏦 Loan Default Prediction System")
st.write("Predict whether a customer will default on a loan using Machine Learning.")

# Load trained model
model = joblib.load("models/best_model.pkl")

# User Inputs
loan_amount = st.number_input("Loan Amount", min_value=10000)
income = st.number_input("Annual Income", min_value=10000)
credit_score = st.slider("Credit Score", 300, 850, 650)
loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72])
existing_loans = st.slider("Existing Loans", 0, 5, 0)

if st.button("Predict"):

    input_data = np.array([[loan_amount, income, credit_score, loan_term, existing_loans]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ High Risk: Loan Default Likely")
    else:
        st.success("✅ Low Risk: Loan Default Unlikely")
