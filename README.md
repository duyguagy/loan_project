# Loan Approval Prediction Project

## 📌 Project Overview
This project aims to predict whether a loan application will be approved or not based on applicant financial and personal information. Machine learning models are used to analyze patterns in the dataset and make predictions.

---

## 📊 Dataset
The dataset used in this project is `loan_approval_dataset.csv`.

It contains information about loan applicants such as:
- loan_id
- no_of_dependents
- income_annum
- loan_amount
- loan_term
- cibil_score
- education	self_employed 
- residential_assets_value
- commercial_assets_value 
- luxury_assets_value	
- bank_asset_value	
- loan_status

**Target variable:** loan_status (Approved / Not Approved)

---

## 🧠 Methodology

### 1. Column Audit
Irrelevant or redundant columns were identified and removed to avoid data leakage and improve model performance.

---

### 2. Exploratory Data Analysis (EDA)
EDA was performed to understand the dataset:
- Missing values check
- Distribution of numerical features
- Relationship between features and target variable
- Correlation analysis

---

### 3. Feature Engineering
New features were created from existing data where necessary (e.g. log transformations or ratio-based features).

---

### 4. Feature Selection
Features with low importance or high correlation were removed to reduce noise and improve model accuracy.

---

### 5. Preprocessing Pipeline
A full preprocessing pipeline was built using `sklearn`:
- Missing value imputation
- Encoding categorical variables
- Scaling numerical features
- ColumnTransformer + Pipeline integration

---

### 6. Model Training & Evaluation
Multiple machine learning models were trained and compared.

Models evaluated include:
- Logistic Regression
- Decision Tree
- Random Forest (or other models you tested)

The best model was selected based on performance on validation data.

---

## 📈 Results
- Train Score: (1.0)
- Test Score: (0.9988290398126464)
- Evaluation Metric: Accuracy / F1-score

Best performing model: (write model name)

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
streamlit run app.py