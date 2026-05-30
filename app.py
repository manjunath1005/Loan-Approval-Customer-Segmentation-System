import streamlit as st
import joblib
import pandas as pd

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>
.main .block-container{
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 95%;
}

.metric-card {
    padding: 1rem;
    border-radius: 10px;
    background-color: #f5f5f5;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_models():

    loan_model = joblib.load(
        "models/loan_approval_model.pkl"
    )

    loan_scaler = joblib.load(
        "models/loan_scaler.pkl"
    )

    segment_model = joblib.load(
        "models/customer_segmentation_model.pkl"
    )

    segment_scaler = joblib.load(
        "models/segmentation_scaler.pkl"
    )

    pca_model = joblib.load(
        "models/pca_model.pkl"
    )

    return (
        loan_model,
        loan_scaler,
        segment_model,
        segment_scaler,
        pca_model
    )

(
    loan_model,
    loan_scaler,
    segment_model,
    segment_scaler,
    pca_model
) = load_models()

# =====================================
# SIDEBAR
# =====================================

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "💰 Loan Approval Prediction",
        "👥 Customer Segmentation",
        "📊 PCA Information",
        "ℹ️ About Project"
    ]
)

# =====================================
# HOME PAGE
# =====================================

if page == "🏠 Home":

    st.title(
        "🏦 AI-Driven Loan Approval & Customer Segmentation System"
    )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Dataset Records",
            "255,347"
        )

    with c2:
        st.metric(
            "Features",
            "16"
        )

    with c3:
        st.metric(
            "Algorithms",
            "3"
        )

    with c4:
        st.metric(
            "Default Recall",
            "70%"
        )

    st.markdown("---")

    st.subheader("📌 Project Modules")

    st.markdown("""
### 💰 Loan Approval Prediction
Predicts whether a customer is likely to default on a loan using Logistic Regression.

### 👥 Customer Segmentation
Groups customers into meaningful business segments using K-Means Clustering.

### 📊 PCA Analysis
Uses Principal Component Analysis for dimensionality reduction and visualization.

### 🎯 Business Objective
Assist financial institutions in reducing loan default risk and improving customer profiling.
""")

    st.success("System Ready For Predictions")

# =====================================
# LOAN APPROVAL PAGE
# =====================================

elif page == "💰 Loan Approval Prediction":

    st.title("💰 Loan Approval Prediction")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            18,
            100,
            30
        )

        income = st.number_input(
            "Income",
            min_value=0.0,
            value=50000.0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=100000.0
        )

        credit_score = st.number_input(
            "Credit Score",
            300,
            850,
            650
        )

        months_employed = st.number_input(
            "Months Employed",
            min_value=0,
            value=24
        )

        num_credit_lines = st.number_input(
            "Number of Credit Lines",
            min_value=0,
            value=3
        )

        interest_rate = st.number_input(
            "Interest Rate",
            min_value=0.0,
            value=10.0
        )

        loan_term = st.number_input(
            "Loan Term",
            min_value=12,
            value=36
        )

    with col2:

        dti_ratio = st.slider(
            "DTI Ratio",
            0.0,
            1.0,
            0.30
        )

        education = st.selectbox(
            "Education",
            ["Bachelor's", "High School", "Master's", "PhD"]
        )

        employment = st.selectbox(
            "Employment Type",
            [
                "Full-time",
                "Part-time",
                "Self-employed",
                "Unemployed"
            ]
        )

        marital = st.selectbox(
            "Marital Status",
            [
                "Divorced",
                "Married",
                "Single"
            ]
        )

        mortgage = st.selectbox(
            "Has Mortgage",
            ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Has Dependents",
            ["No", "Yes"]
        )

        purpose = st.selectbox(
            "Loan Purpose",
            [
                "Auto",
                "Business",
                "Education",
                "Home",
                "Other"
            ]
        )

        cosigner = st.selectbox(
            "Has Co-Signer",
            ["No", "Yes"]
        )

    if st.button("Predict Loan Status"):

        education_map = {
            "Bachelor's": 0,
            "High School": 1,
            "Master's": 2,
            "PhD": 3
        }

        employment_map = {
            "Full-time": 0,
            "Part-time": 1,
            "Self-employed": 2,
            "Unemployed": 3
        }

        marital_map = {
            "Divorced": 0,
            "Married": 1,
            "Single": 2
        }

        yes_no_map = {
            "No": 0,
            "Yes": 1
        }

        purpose_map = {
            "Auto": 0,
            "Business": 1,
            "Education": 2,
            "Home": 3,
            "Other": 4
        }

        input_data = [[
            age,
            income,
            loan_amount,
            credit_score,
            months_employed,
            num_credit_lines,
            interest_rate,
            loan_term,
            dti_ratio,
            education_map[education],
            employment_map[employment],
            marital_map[marital],
            yes_no_map[mortgage],
            yes_no_map[dependents],
            purpose_map[purpose],
            yes_no_map[cosigner]
        ]]

        scaled_data = loan_scaler.transform(
            input_data
        )

        prediction = loan_model.predict(
            scaled_data
        )[0]

        probs = loan_model.predict_proba(
            scaled_data
        )[0]

        safe_prob = probs[0]
        default_prob = probs[1]

        st.markdown("---")

        if prediction == 0:
            st.success("✅ LOAN APPROVED")
        else:
            st.error("❌ LOAN REJECTED")

        colA, colB = st.columns(2)

        with colA:
            st.metric(
                "Default Probability",
                f"{default_prob*100:.2f}%"
            )

        with colB:
            st.metric(
                "Safe Probability",
                f"{safe_prob*100:.2f}%"
            )

        st.subheader("Probability Visualization")

        st.write("Default Probability")
        st.progress(float(default_prob))

        st.write("Safe Probability")
        st.progress(float(safe_prob))

        if default_prob < 0.20:
            risk = "🟢 Low Risk"
            st.success(f"Risk Category: {risk}")

        elif default_prob < 0.50:
            risk = "🟡 Medium Risk"
            st.warning(f"Risk Category: {risk}")

        else:
            risk = "🔴 High Risk"
            st.error(f"Risk Category: {risk}")

        st.subheader("Business Recommendation")

        if default_prob < 0.20:
            st.success(
                "Customer appears financially stable and is likely eligible for loan approval."
            )

        elif default_prob < 0.50:
            st.warning(
                "Customer falls into a moderate-risk category. Manual review is recommended."
            )

        else:
            st.error(
                "Customer shows a high probability of default. Loan approval should be reconsidered."
            )

# =====================================
# CUSTOMER SEGMENTATION
# =====================================

elif page == "👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    age = st.number_input(
        "Age",
        18,
        100,
        30,
        key="seg_age"
    )

    income = st.number_input(
        "Income",
        min_value=0.0,
        value=50000.0,
        key="seg_income"
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=100000.0,
        key="seg_loan"
    )

    credit_score = st.number_input(
        "Credit Score",
        300,
        850,
        650,
        key="seg_credit"
    )

    dti_ratio = st.slider(
        "DTI Ratio",
        0.0,
        1.0,
        0.30,
        key="seg_dti"
    )

    if st.button("Find Customer Segment"):

        seg_data = [[
            income,
            credit_score,
            loan_amount,
            dti_ratio,
            age
        ]]

        seg_scaled = segment_scaler.transform(
            seg_data
        )

        cluster = segment_model.predict(
            seg_scaled
        )[0]

        labels = {
            0: "Premium Customer",
            1: "High Income Risk",
            2: "High Risk Customer"
        }

        segment = labels[cluster]

        if cluster == 0:
            st.success(
                f"⭐ Customer Segment: {segment}"
            )

        elif cluster == 1:
            st.warning(
                f"⚠️ Customer Segment: {segment}"
            )

        else:
            st.error(
                f"🚨 Customer Segment: {segment}"
            )

        segment_info = {
            "Premium Customer":
                "High credit score and financially stable customer.",

            "High Income Risk":
                "High income but weaker credit profile.",

            "High Risk Customer":
                "Lower creditworthiness and higher default risk."
        }

        st.info(segment_info[segment])

# =====================================
# PCA PAGE
# =====================================

elif page == "📊 PCA Information":

    st.title("📊 PCA Analysis")

    st.markdown("""
### Principal Component Analysis (PCA)

PCA was applied to reduce dimensionality and visualize customer clusters.

### Results

- PC1 Variance = 20.12%
- PC2 Variance = 20.05%
- Total Variance Retained = 40.16%

### Benefits

✅ Reduced dimensionality

✅ Better visualization

✅ Faster analysis

✅ Improved cluster interpretation
""")

# =====================================
# ABOUT PAGE
# =====================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Project")

    st.markdown("""
## AI-Driven Loan Approval & Customer Segmentation System

### Business Problem

Banks need to determine whether a loan applicant is likely to repay or default.

This system assists financial institutions by providing:

- Loan Approval Prediction
- Customer Segmentation
- Risk Assessment
- Data Visualization

---

### Dataset Information

- Total Records: 255,347
- Features: 16
- Target Variable: Default

---

### Machine Learning Models

#### Logistic Regression
- Accuracy: 67.4%
- Default Recall: 70%

#### K-Means Clustering
- Premium Customer
- High Income Risk
- High Risk Customer

#### PCA
- Variance Retained: 40.16%

---

### Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Joblib

""")