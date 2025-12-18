# 💳 Loan Risk Classification with Explainable AI

An end-to-end **Credit / Loan Risk Classification system** that predicts whether a loan applicant is **Safe** or **Risky** using machine learning, with a strong focus on **model explainability (XAI)** and real-world deployment via **Streamlit**.

---

## 📌 Project Overview

Financial institutions must evaluate loan applicants accurately while maintaining transparency.  
This project addresses that need by combining:

- **XGBoost-based supervised learning**
- **Robust preprocessing pipelines**
- **Explainable AI (SHAP + rule-based reasoning)**
- **Interactive Streamlit dashboard**

The system outputs both a **risk probability** and **clear reasons behind the decision**, making it suitable for real-world financial decision support.

---

## 🎯 Objectives

- Classify loan applicants as **Safe (0)** or **Risky (1)**
- Handle mixed numerical and categorical features
- Address class imbalance
- Ensure transparency using Explainable AI
- Deploy a user-friendly web interface

---

## 📂 Project Structure
Loan-Risk-Classification/
│
├── data/
│ └── credit_risk_dataset.csv
│
├── notebooks/
│ └── credit_risk_analysis.ipynb
│
├── model/
│ └── xgb_model.pkl
│
├── app.py
├── requirements.txt
└── README.md