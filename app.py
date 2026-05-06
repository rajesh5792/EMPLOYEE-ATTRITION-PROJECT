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

df = pd.read_csv("HR-Employee-Attrition.csv")

# Page config
icon = Image.open("logo.png")
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon=icon,
    layout="wide"
)

# =========================================
# TITLE SECTION
# =========================================

st.markdown("""
# Employee Attrition Prediction Dashboard
""")

st.caption(
    "AI-powered workforce retention analytics and employee risk prediction system"
)

st.markdown("---")

# =========================================
# SESSION STATE
# =========================================

if "page" not in st.session_state:
    st.session_state.page = "About"

# =========================================
# SIDEBAR NAVIGATION
# =========================================

with st.sidebar:

    st.title("Navigation")

    if st.button("About"):
        st.session_state.page = "About"

    if st.button("Model Info"):
        st.session_state.page = "Model Info"

    if st.button("Prediction Dashboard"):
        st.session_state.page = "Dashboard"

# =========================================
# ABOUT PAGE
# =========================================

if st.session_state.page == "About":

    st.header("About the Dashboard")

    st.write("""
    The Employee Attrition Prediction Dashboard is an AI-powered workforce analytics
system developed to predict the likelihood of employees leaving an organization
based on workforce-related behavioral and organizational factors.

Employee attrition is one of the major challenges faced by organizations,
as high employee turnover can negatively impact productivity, operational stability,
employee morale, recruitment costs, and overall organizational growth.

This system uses Machine Learning techniques to analyze important employee-related
factors such as monthly income, work-life balance, overtime, job satisfaction,
age, and years at the company to estimate employee attrition probability
and workforce retention risk.
    """)

    st.markdown("---")


    # =========================================
    # BUSINESS IMPACT
    # =========================================

    st.subheader("Why Employee Attrition Matters")

    st.write("""
    High employee turnover affects multiple organizational factors including:
    
    - Organizational productivity
    - Recruitment costs
    - Employee morale
    - Operational efficiency
    - Long-term workforce stability
    
    This project was developed to analyze workforce-related factors
    and predict employee attrition probability using Machine Learning.
    """)

    st.markdown("---")

    # =========================================
    # SYSTEM OBJECTIVES
    # =========================================

    st.subheader("System Objectives")

    st.write("""
    The primary objective of this system is to help organizations
    proactively identify employees who may show signs of attrition risk.
    
    The dashboard aims to:
    
    - Predict employee attrition probability
    - Classify employee retention risk levels
    - Analyze workforce-related employee behavior
    - Provide workforce visualizations and analytics
    - Generate employee retention recommendations
    - Support HR analytics and strategic workforce planning
    
    By combining Machine Learning with interactive analytics,
    the system enables organizations to make informed workforce
    management decisions and improve employee retention strategies.
    """)

    st.markdown("---")

    # =========================================
    # KEY FEATURES
    # =========================================

    st.subheader("Key Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:

        st.success("Attrition Probability Prediction")

        st.success("Risk Classification System")

        st.success("Employee Position Analysis")

    with feature_col2:

        st.success("Interactive Workforce Visualizations")

        st.success("Retention Recommendation System")

        st.success("Real-time Dashboard Interaction")

    st.markdown("---")

    # =========================================
    # PROJECT GOAL
    # =========================================

    st.subheader("Project Goal")

    st.info("""
    The goal of this system is to help organizations identify workforce retention risks,
    improve employee satisfaction, and support data-driven HR decision-making.
    """) 

# =========================================
# MODEL INFO PAGE
# =========================================

elif st.session_state.page == "Model Info":

    st.header("Model Information")

    st.subheader("Machine Learning Model")

    st.write("""
    This dashboard uses a Logistic Regression Machine Learning model trained on HR analytics data
to predict employee attrition probability based on workforce-related behavioral and organizational factors.

The model analyzes employee-related information such as monthly income, job satisfaction,
work-life balance, overtime, age, and years at the company to estimate employee retention risk
and classify workforce stability patterns.

The prediction system is designed to provide probability-based attrition analysis,
risk categorization, workforce insights, and employee retention recommendations
through an interactive analytics dashboard.
             
Logistic Regression was selected because:

 - It provides interpretable probability outputs
 - It performs effectively on structured HR datasets
 - It produces stable prediction behavior
 - It is computationally efficient
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
        label="Model Accuracy",
        value="86.3%"
    )

    with col2:
        st.metric(
        label="ML Algorithm",
        value="Logistic Regression"
    )

    with col3:
        st.metric(
        label="Features Used",
        value="6"
    )

    st.markdown("---")


    st.subheader("Features Used for Prediction")

    st.write("""
    The following employee-related features are used:
    
    - Age
    - Monthly Income
    - Years at Company
    - Job Satisfaction
    - Work Life Balance
    - Overtime
    """)
    st.markdown("---")

    st.subheader("Prediction Workflow")

    st.write("""
    The prediction process follows these steps:
    
    1. Employee information is collected through dashboard inputs.
    
    2. Categorical values are converted into numerical representations.
    
    3. The processed data is passed into the trained Logistic Regression model.
    
    4. The model calculates employee attrition probability.
    
    5. Probability thresholds are used to classify:
    
       - Low Risk
       - Medium Risk
       - High Risk
    
    6. The system generates:
    
       - Risk analysis
       - Workforce visualizations
       - Retention recommendations
    """)

# =========================================
# PREDICTION DASHBOARD
# =========================================

elif st.session_state.page == "Dashboard":

    st.header("Employee Attrition Prediction")

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
    
    # =========================================
    # PREDICT BUTTON
    # =========================================

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

        # =========================================
        # RECOMMENDATION SECTION
        # =========================================

        st.markdown("---")

        st.subheader("Employee Retention Recommendations")

        recommendations = []

        if overtime == "Yes":
            recommendations.append(
                "Reduce overtime workload to improve employee retention."
            )

        if work_life <= 2:
            recommendations.append(
                "Improve work-life balance policies and flexibility."
            )

        if job_satisfaction <= 2:
            recommendations.append(
                "Employee engagement and recognition programs may help improve satisfaction."
            )

        if years < 2:
            recommendations.append(
                "New employees require stronger onboarding and mentorship support."
            )

        if income < 5000:
            recommendations.append(
                "Low monthly income may contribute to higher attrition risk."
            )

        if age < 25:
            recommendations.append(
                "Younger employees often show higher job-switch tendencies."
            )

        if len(recommendations) == 0:

            st.success(
                "No major employee attrition indicators detected."
            )

        else:

            for rec in recommendations:
                st.warning(rec)