import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load model
with open('model.pkl', 'rb') as f:
    lr = pickle.load(f)

df = pd.read_csv(r"C:\Users\logan\OneDrive\Desktop\Employee Attrition project\HR-Employee-Attrition.csv")

# Page config
icon = Image.open(r"C:\Users\logan\OneDrive\Desktop\Employee Attrition project\logo.png")
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon=icon,
    layout="wide"
)

# PREMIUM TITLE SECTION HERE

st.markdown("""
# Employee Attrition Prediction Dashboard

AI-powered workforce retention analytics system
""")


# SIDEBAR HERE
with st.sidebar:
    st.header("Dashboard Info")

    st.write("""
    This dashboard predicts employee attrition risk
    using machine learning techniques.
    """)

    st.header("About")

    st.write("""
    This system predicts the likelihood of employee attrition
    using machine learning techniques trained on HR analytics data.
    """)

    st.subheader("Model Information")

    st.write("""
    Model: Logistic Regression
    
    Accuracy: 86.3%
    
    Features Used:
    - Age
    - Monthly Income
    - Years at Company
    - Job Satisfaction
    - Work Life Balance
    - Overtime
    """)

# MAIN TITLE
st.title("Employee Attrition Prediction System")

st.caption("AI-powered workforce retention analysis")

# remaining UI...

st.divider()


st.markdown("## Employee Information")

col1, col2, col3 = st.columns(3)

# ---------- COLUMN 1 ----------
with col1:

    age = st.slider(
        "Age",
        18,
        60,
        30
    )

    years = st.slider(
        "Years at Company",
        0,
        40,
        5
    )

# ---------- COLUMN 2 ----------
with col2:

    income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=500
    )

    overtime = st.selectbox(
        "OverTime",
        ["No", "Yes"]
    )

# ---------- COLUMN 3 ----------
with col3:

    job_satisfaction = st.slider(
        "Job Satisfaction",
        1,
        4,
        2
    )

    st.caption("1 = Low | 4 = High")

    work_life = st.slider(
        "Work Life Balance",
        1,
        4,
        2
    )

    st.caption("1 = Poor | 4 = Excellent")

st.divider()

# Predict button
if st.button("Predict Attrition"):

    overtime_yes = 1 if overtime == "Yes" else 0

    input_data = pd.DataFrame([{
        'Age': age,
        'MonthlyIncome': income,
        'YearsAtCompany': years,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life,
        'OverTime_Yes': overtime_yes
    }])

    proba = lr.predict_proba(input_data)[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

# ---------- METRICS ----------
    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.metric(
            label="Attrition Probability",
            value=f"{proba*100:.2f}%"
        )

    with metric_col2:

        if proba > 0.20:
            risk_level = "High Risk"

        elif proba > 0.10:
            risk_level = "Medium Risk"

        else:
            risk_level = "Low Risk"

        st.metric(
            label="Risk Level",
            value=risk_level
        )

# ---------- PROGRESS BAR ----------
    st.progress(float(proba))

# ---------- RISK MESSAGE ----------
    if proba > 0.20:

        st.error(
            "Employee shows a high likelihood of attrition."
        )

    elif proba > 0.10:

        st.warning(
            "Employee shows a moderate likelihood of attrition."
        )

    else:

        st.success(
            "Employee shows a low likelihood of attrition."
        )



# =========================================
# EMPLOYEE POSITION ANALYSIS
# =========================================

st.markdown("---")

st.subheader("Employee Position Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Age",
    "Income",
    "Years",
    "Satisfaction",
    "Work-Life"
])

# =========================================
# AGE TAB
# =========================================

with tab1:

    fig1, ax1 = plt.subplots(figsize=(4,2))

    sns.histplot(
        data=df,
        x='Age',
        hue='Attrition',
        kde=True,
        bins=30,
        ax=ax1
    )

    ax1.axvline(
        age,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Current Employee'
    )

    ax1.set_title("Employee Age Position")
    ax1.title.set_size(10)

    ax1.tick_params(labelsize=8)

    ax1.legend(fontsize=7)

    graph_col1, graph_col2, graph_col3 = st.columns([1,2,1])

    with graph_col2:
        st.pyplot(fig1, use_container_width=False)

    age_percentile = (df['Age'] < age).mean() * 100

    st.info(
        f"This employee is older than {age_percentile:.1f}% of employees."
    )

# =========================================
# INCOME TAB
# =========================================

with tab2:

    fig2, ax2 = plt.subplots(figsize=(4,2))

    sns.histplot(
        data=df,
        x='MonthlyIncome',
        hue='Attrition',
        kde=True,
        bins=30,
        ax=ax2
    )

    ax2.axvline(
        income,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Current Employee'
    )

    ax2.set_title("Employee Monthly Income Position")
    ax2.title.set_size(10)

    ax2.tick_params(labelsize=8)

    ax2.legend(fontsize=7)

    graph_col1, graph_col2, graph_col3 = st.columns([1,2,1])

    with graph_col2:
        st.pyplot(fig2, use_container_width=False)

    income_percentile = (
        (df['MonthlyIncome'] < income).mean()
    ) * 100

    st.info(
        f"This employee earns more than {income_percentile:.1f}% of employees."
    )

# =========================================
# YEARS TAB
# =========================================

with tab3:

    fig3, ax3 = plt.subplots(figsize=(4,2))

    sns.histplot(
        data=df,
        x='YearsAtCompany',
        hue='Attrition',
        kde=True,
        bins=20,
        ax=ax3
    )

    ax3.axvline(
        years,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Current Employee'
    )

    ax3.set_title("Years at Company Position")
    ax3.title.set_size(10)

    ax3.tick_params(labelsize=8)

    ax3.legend(fontsize=7)

    graph_col1, graph_col2, graph_col3 = st.columns([1,2,1])

    with graph_col2:
        st.pyplot(fig3, use_container_width=False)

    years_percentile = (
        (df['YearsAtCompany'] < years).mean()
    ) * 100

    st.info(
        f"This employee has stayed longer than {years_percentile:.1f}% of employees."
    )

# =========================================
# JOB SATISFACTION TAB
# =========================================

with tab4:

    fig4, ax4 = plt.subplots(figsize=(3.5,2))

    sns.countplot(
        x='JobSatisfaction',
        hue='Attrition',
        data=df,
        ax=ax4
    )

    ax4.axvline(
        job_satisfaction - 1,
        color='red',
        linestyle='--',
        linewidth=2
    )

    ax4.set_title("Job Satisfaction Position")
    ax4.title.set_size(10)

# Reduce axis label size
    ax4.set_xlabel("JobSatisfaction", fontsize=8)
    ax4.set_ylabel("Count", fontsize=8)

# Reduce tick size
    ax4.tick_params(axis='both', labelsize=7)

# Reduce legend size
    ax4.legend(title='Attrition', fontsize=7, title_fontsize=8)

    graph_col1, graph_col2, graph_col3 = st.columns([1,2,1])

    with graph_col2:
        st.pyplot(fig4, use_container_width=False)
# =========================================
# WORK LIFE BALANCE TAB
# =========================================

with tab5:

    fig5, ax5 = plt.subplots(figsize=(3.5,2))

    sns.countplot(
        x='WorkLifeBalance',
        hue='Attrition',
        data=df,
        ax=ax5
    )

    ax5.axvline(
        work_life - 1,
        color='red',
        linestyle='--',
        linewidth=2
    )

    ax5.set_title("Work Life Balance Position")
    ax5.title.set_size(10)

# Reduce axis label size
    ax5.set_xlabel("WorkLifeBalance", fontsize=8)
    ax5.set_ylabel("Count", fontsize=8)

# Reduce tick size
    ax5.tick_params(axis='both', labelsize=7)

# Reduce legend size
    ax5.legend(title='Attrition', fontsize=7, title_fontsize=8)
    graph_col1, graph_col2, graph_col3 = st.columns([1,2,1])

    with graph_col2:
        st.pyplot(fig5, use_container_width=False)
    
# =========================================
# RECOMMENDATION SECTION
# =========================================

st.markdown("---")

st.subheader("Employee Retention Recommendations")

recommendations = []

# Overtime check
if overtime == "Yes":
    recommendations.append(
        "Reduce overtime workload to improve employee retention."
    )

# Work-life balance check
if work_life <= 2:
    recommendations.append(
        "Improve work-life balance policies and flexibility."
    )

# Job satisfaction check
if job_satisfaction <= 2:
    recommendations.append(
        "Employee engagement and recognition programs may help improve satisfaction."
    )

# Experience check
if years < 2:
    recommendations.append(
        "New employees require stronger onboarding and mentorship support."
    )

# Income check
if income < 5000:
    recommendations.append(
        "Low monthly income may contribute to higher attrition risk."
    )

# Age check
if age < 25:
    recommendations.append(
        "Younger employees often show higher job-switch tendencies."
    )

# Final output
if len(recommendations) == 0:

    st.success(
        "No major employee attrition indicators detected."
    )

else:

    for rec in recommendations:
        st.warning(rec)