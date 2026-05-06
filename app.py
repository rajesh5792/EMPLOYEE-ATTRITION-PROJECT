import streamlit as st 
import pickle
import numpy as np
with open('model.pkl', 'rb') as f:
    lr = pickle.load(f)

# Page config
st.set_page_config(page_title="Attrition Predictor", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>Employee Attrition Prediction</h1>", unsafe_allow_html=True)
st.write("Predict whether an employee is likely to leave the company.")

st.divider()

st.subheader("Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 30)

    years = st.slider("Years at Company", 0, 40, 5)

    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2)

with col2:
    income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=50000,
        value=10000
    )

    work_life = st.slider("Work Life Balance", 1, 4, 2)

    overtime = st.selectbox("OverTime", ["Yes", "No"])

st.divider()

# Predict button
if st.button("Predict Attrition"):

    overtime_yes = 1 if overtime == "Yes" else 0

    import pandas as pd

    input_data = pd.DataFrame([{
        'Age': age,
        'MonthlyIncome': income,
        'YearsAtCompany': years,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life,
        'OverTime_Yes': overtime_yes
    }])

    proba = lr.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Attrition Probability: {proba:.2%}")
    if proba > 0.20:
        st.error(f"High Attrition Risk ({proba:.2%})")

    elif proba > 0.10:
        st.warning(f"Medium Attrition Risk ({proba:.2%})")

    else:
        st.success(f"Low Attrition Risk ({proba:.2%})")