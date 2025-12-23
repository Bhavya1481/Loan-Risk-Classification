# 💳 Loan Risk Classification with Explainable AI (XAI)

An end-to-end **Credit / Loan Risk Classification System** that predicts whether a loan applicant is **Safe** or **Risky** using machine learning, while ensuring **model transparency and explainability** through rule-based logic and SHAP analysis.  
The project also includes an **interactive Streamlit dashboard** for real-time predictions and visualization.

---

## 📌 Problem Statement

Financial institutions must evaluate loan applications accurately to minimize default risk while maintaining **transparent decision-making**.  
Traditional models often act as black boxes, making it difficult to justify why an applicant is classified as risky.

This project solves that problem by:
- Building a **high-performance ML model**
- Handling **real-world financial data challenges**
- Providing **Explainable AI (XAI)** to justify predictions

---

## 🎯 Project Objectives

- Predict whether a loan applicant is **Safe (0)** or **Risky (1)**
- Handle mixed numerical and categorical features
- Address class imbalance in credit data
- Achieve high recall for risky applicants
- Provide **human-readable explanations** for predictions
- Deploy a user-friendly web application

---

---

## 🧾 Prerequisites

Before running this project, ensure you have the following installed:

- **Python 3.9 – 3.10 (recommended)**
- **pip (Python package manager)**
- **Git**
- A modern web browser (Chrome / Firefox)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Bhavya1481/Loan-Risk-Classification.git
cd Loan-Risk-Classification
````

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python3 -m venv credit_ml
```

Activate the environment:

* **macOS / Linux**

```bash
source credit_ml/bin/activate
```

* **Windows**

```bash
credit_ml\Scripts\activate
```

---

### 3️⃣ Install Required Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Dataset Description

The dataset contains **demographic, employment, and financial attributes** of loan applicants.

### 🎯 Target Variable

* `loan_status`

  * `0` → Safe Applicant
  * `1` → Risky Applicant

### 📌 Input Features

| Feature                    | Description               |
| -------------------------- | ------------------------- |
| person_age                 | Applicant age             |
| person_income              | Annual income             |
| person_home_ownership      | RENT / OWN / MORTGAGE     |
| person_emp_length          | Employment length (years) |
| loan_intent                | Purpose of loan           |
| loan_grade                 | Credit grade (A–G)        |
| loan_amnt                  | Loan amount               |
| loan_int_rate              | Interest rate             |
| loan_percent_income        | Loan burden               |
| cb_person_default_on_file  | Previous default          |
| cb_person_cred_hist_length | Credit history length     |

---

## ⚙️ Data Preprocessing

* Missing value handling using **median & most-frequent imputation**
* Categorical encoding via **One-Hot Encoding**
* Numerical feature scaling using **StandardScaler**
* Feature engineering:

loan_percent_income = loan_amount / annual_income

* Data leakage prevention using **Pipelines**
* Class imbalance handled using **SMOTE**

---

## 🧠 Machine Learning Model

### 🔹 Algorithm Used

* **XGBoost Classifier**

### 🔹 Mathematical Formulation

z = ΣFk(x)

P(Risky) = 1/1+e^(-z)

### 🔹 Decision Rule

If P ≥ 0.5 ⇒ Risky
Else ⇒ Safe

---

## 📈 Model Evaluation

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC–AUC Curve
* Confusion Matrix

⚠️ **Recall for risky applicants is prioritized** to minimize financial loss.

---

## 🔍 Explainable AI (XAI)

### 1️⃣ Rule-Based Explainability (Online)

Displayed in Streamlit using domain rules:

* High loan-to-income ratio
* High interest rate
* Short credit history
* Previous default record
* Low employment stability

---

### 2️⃣ SHAP Explainability (Offline)

* SHAP values computed **offline** using `KernelExplainer`
* Explains global feature importance
* Avoids runtime instability
* SHAP plots displayed in the dashboard

---

## 🌐 Streamlit Web Application

### 🔹 Features

* Interactive input form
* Real-time credit risk prediction
* Probability-based classification
* Rule-based explanations
* SHAP feature importance visualization

### ▶️ Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries & Tools

* XGBoost
* Scikit-learn
* Pandas, NumPy
* SHAP (offline)
* Streamlit
* Matplotlib, Seaborn
* Joblib

---

