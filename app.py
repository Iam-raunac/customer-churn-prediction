# import streamlit as st
# import pandas as pd
# import pickle

# # load model data
# with open("customer_churn_model.pkl", "rb") as f:
#     data = pickle.load(f)

# model = data["model"]
# features = data["features"]
# encoders = data["encoders"]

# st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# st.title("📉 Customer Churn Prediction App")

# st.write("Fill the customer details to predict churn")

# # -------- INPUT FORM --------
# gender = st.selectbox("Gender", ["Male", "Female"])
# SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
# Partner = st.selectbox("Partner", ["Yes", "No"])
# Dependents = st.selectbox("Dependents", ["Yes", "No"])
# tenure = st.number_input("Tenure (months)", min_value=0, max_value=72)
# PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
# MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
# InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
# OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
# OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
# DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
# TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
# StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
# StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
# Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
# PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
# PaymentMethod = st.selectbox(
#     "Payment Method",
#     ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
# )
# MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
# TotalCharges = st.number_input("Total Charges", min_value=0.0)

# # -------- CREATE INPUT DF --------
# input_data = {
#     "gender": gender,
#     "SeniorCitizen": SeniorCitizen,
#     "Partner": Partner,
#     "Dependents": Dependents,
#     "tenure": tenure,
#     "PhoneService": PhoneService,
#     "MultipleLines": MultipleLines,
#     "InternetService": InternetService,
#     "OnlineSecurity": OnlineSecurity,
#     "OnlineBackup": OnlineBackup,
#     "DeviceProtection": DeviceProtection,
#     "TechSupport": TechSupport,
#     "StreamingTV": StreamingTV,
#     "StreamingMovies": StreamingMovies,
#     "Contract": Contract,
#     "PaperlessBilling": PaperlessBilling,
#     "PaymentMethod": PaymentMethod,
#     "MonthlyCharges": MonthlyCharges,
#     "TotalCharges": TotalCharges
# }

# input_df = pd.DataFrame([input_data])

# # encode categorical columns
# for col, encoder in encoders.items():
#     input_df[col] = encoder.transform(input_df[col])

# # reorder columns
# input_df = input_df[features]

# # -------- PREDICT --------
# if st.button("Predict Churn"):
#     prediction = model.predict(input_df)
#     probability = model.predict_proba(input_df)[0][1]

#     if prediction[0] == 1:
#         st.error(f"❌ Customer is likely to CHURN (Probability: {probability:.2f})")
#     else:
#         st.success(f"✅ Customer is NOT likely to churn (Probability: {probability:.2f})")

import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
with open("customer_churn_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
features = data["features"]
encoders = data["encoders"]

# ---------------- UI HEADER ----------------
st.title("📉 Customer Churn Prediction App")
st.write(
    "This application predicts whether a telecom customer is likely to churn "
    "based on their service usage and billing information."
)

st.markdown("---")

# ---------------- INPUT FORM ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox(
    "Multiple Lines", ["Yes", "No", "No phone service"]
)
InternetService = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
Contract = st.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)
MonthlyCharges = st.number_input(
    "Monthly Charges", min_value=0.0, max_value=500.0, value=70.0
)
TotalCharges = st.number_input(
    "Total Charges", min_value=0.0, value=1000.0
)

# ---------------- ADVANCED CONTROL ----------------
st.markdown("### ⚙️ Prediction Settings")
threshold = st.slider(
    "Churn Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    help="Adjust sensitivity of churn prediction",
)

# ---------------- CREATE INPUT DF ----------------
input_data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}

input_df = pd.DataFrame([input_data])

# ---------------- ENCODE ----------------
for col, encoder in encoders.items():
    input_df[col] = encoder.transform(input_df[col])

input_df = input_df[features]

# ---------------- PREDICT ----------------
if st.button("🔮 Predict Churn"):
    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= threshold)

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.progress(prob)
    st.write(f"**Churn Probability:** `{prob:.2f}`")

    if prediction == 1:
        st.error("❌ Customer is **LIKELY TO CHURN**")
    else:
        st.success("✅ Customer is **NOT LIKELY TO CHURN**")