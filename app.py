import streamlit as st 
import pickle
import numpy as np
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page config
st.set_page_config(page_title="Attrition Predictor", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>Employee Attrition Prediction</h1>", unsafe_allow_html=True)
st.write("Predict whether an employee is likely to leave the company.")

st.divider()

# Input section
st.subheader("Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60)
    years = st.slider("Years at Company", 0, 40)

with col2:
    income = st.number_input("Monthly Income", min_value=1000, max_value=20000)

st.divider()

# Predict button
if st.button("Predict Attrition"):
    input_data = np.array([[age, income, years]])
    proba = model.predict_proba(input_data)[0][1]
    st.subheader("Prediction Result")

    if proba > 0.4:
        st.error("🔴 High Risk")
    elif proba > 0.25:
        st.warning("🟠 Medium Risk")
    else:
        st.success("🟢 Low Risk")