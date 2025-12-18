import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Risk Prediction System")
st.write("Predict whether a loan applicant is **Safe** or **Risky** and understand the reasons behind the decision.")


# LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load("model/xgb_model.pkl")

model = load_model()

# USER INPUTS
st.header("📝 Applicant Information")

age = st.slider("Age", 18, 70, 30)
income = st.number_input("Annual Income", min_value=1000, value=50000)
employment_length = st.slider("Employment Length (years)", 0, 40, 5)

home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_intent = st.selectbox(
    "Loan Purpose",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

loan_amount = st.number_input("Loan Amount", min_value=500, value=10000)
interest_rate = st.slider("Interest Rate (%)", 1.0, 35.0, 12.5)

credit_history = st.slider("Credit History Length (years)", 0, 30, 6)

previous_default = st.selectbox(
    "Previous Default on Record?",
    ["N", "Y"]
)

# PREDICTION

if st.button("🔍 Predict Credit Risk"):

    # Create full input dataframe (MUST MATCH TRAINING FEATURES)
    input_df = pd.DataFrame([{
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home_ownership,
        "person_emp_length": employment_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amount,
        "loan_int_rate": interest_rate,
        "loan_percent_income": loan_amount / income,
        "cb_person_default_on_file": previous_default,
        "cb_person_cred_hist_length": credit_history
    }])

    # Predict probability
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    if prob >= 0.5:
        st.error(f" **Risky Applicant**\n\nRisk Probability: **{prob:.2f}**")
    else:
        st.success(f" **Safe Applicant**\n\nRisk Probability: **{prob:.2f}**")


    # EXPLAINABILITY (RULE-BASED XAI)

    st.subheader("🧠 Reasons Behind the Decision")

    reasons = []

    if loan_amount / income > 0.4:
        reasons.append("High loan amount compared to income")

    if interest_rate > 15:
        reasons.append("High interest rate")

    if credit_history < 5:
        reasons.append("Short credit history")

    if previous_default == "Y":
        reasons.append("Previous loan default on record")

    if employment_length < 2:
        reasons.append("Low employment stability")

    if reasons:
        for r in reasons:
            st.warning(f"• {r}")
    else:
        st.info("No major risk factors detected based on domain rules.")

# FOOTER
st.markdown("---")
st.caption(" Model: XGBoost + Preprocessing Pipeline | Explainability: Rule-based XAI")