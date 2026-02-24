## Fraud Detection Project (Transaction Data)

This repository contains a machine learning project for **proactive fraud detection on financial transactions**, implemented in a Jupyter notebook and powered by a large-scale synthetic transaction dataset.

### What this project does

- **Builds a fraud detection pipeline** from raw CSV data to model evaluation.
- **Handles extreme class imbalance** in transaction labels.
- **Engineers behavioral balance features** that capture suspicious activity.
- **Trains and evaluates a classification model** for real-time fraud screening.
- **Extracts business insights and recommendations** that are actionable for risk teams.

### Repository structure

- `project.ipynb`: main notebook with data exploration, preprocessing, feature engineering, modeling, and evaluation.
- `Fraud.csv`: transaction-level dataset used for training and evaluation.
- `Data Dictionary.txt`: description of columns and their meanings.
- `.gitignore`: ignores common temporary / environment / notebook artifact files.

### Getting started

1. **Set up your environment**
   - Install Python 3.9+.
   - Create and activate a virtual environment (recommended).
   - Install typical data-science dependencies (for example):

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

2. **Launch the notebook**

```bash
jupyter notebook project.ipynb
```

3. **Run the analysis**
   - Step through the cells in order: load data, preprocess, engineer features, train model, and review evaluation metrics and plots.
   - Adjust model hyperparameters or thresholds to explore different precision/recall trade-offs.

### Data and modeling summary (high level)

- **Records**: ~6.3M transactions, with fraud concentrated in `TRANSFER` and `CASH_OUT` types.
- **Problem**: extremely imbalanced classification (fraud rate ≈ 0.13%).
- **Engineered features**: balance differences and balance errors on origin/destination accounts to capture abnormal money movement.
- **Baseline model**: logistic regression with class weighting for imbalance handling.
- **Key metrics**: ROC-AUC ≈ 0.97 with strong precision, making the model suitable as a real-time fraud screen with threshold tuning.

### How to use / extend this project

- **Experiment with models**: swap logistic regression for tree-based models (e.g., Random Forest, XGBoost) and compare metrics.
- **Tune decision thresholds**: optimize probability thresholds for different business risk appetites (high precision vs. high recall).
- **Add monitoring**: log prediction distributions, false positives, and false negatives over time for production use.
- **Integrate with a system**: wrap the preprocessing + model into an API endpoint or batch scoring job for real-world deployment.

---

The following section is the **original detailed project report**, preserved for reference.

📌 Project Overview

This project builds a proactive fraud detection system for financial transactions using machine learning techniques.

The dataset contains over 6.3 million transactions with simulated fraudulent behavior.

🎯 Objectives

Clean and preprocess large-scale transaction data

Handle extreme class imbalance

Engineer behavioral fraud features

Train and optimize classification model

Extract actionable business insights

Recommend prevention strategies

📂 Dataset Summary
Metric	Value
Total Records	6,362,620
Features	11
Fraud Rate	0.13%
Final Modeling Records	2,770,409

Fraud occurs only in:

TRANSFER

CASH_OUT

🧹 Data Preprocessing

✔ No missing values
✔ No duplicates
✔ Filtered irrelevant transaction types
✔ Removed leakage features
✔ Addressed severe class imbalance

⚙️ Feature Engineering

The following behavioral features were engineered:

orig_balance_diff

dest_balance_diff

orig_balance_error

dest_balance_error

These capture abnormal balance inconsistencies commonly seen in fraudulent transactions.

🤖 Model Development
Algorithm Used:

Logistic Regression (class_weight='balanced')

Why:

Interpretable

Efficient on large datasets

Strong baseline classifier

Handles imbalance well

Data Leakage Removed:

type

isFlaggedFraud

ID columns

📊 Final Model Performance
Metric	Score
ROC-AUC	0.9736
Precision	80%
Recall	48%
F1 Score	0.60
False Positives	196
📉 Confusion Matrix
	Predicted Not Fraud	Predicted Fraud
Actual Not Fraud	552,243	196
Actual Fraud	862	781
🔍 Key Fraud Insights

Fraudulent transactions often show:

Sudden full balance depletion

Large abnormal balance transitions

Mismatch between transaction amount and balance difference

Fraudsters typically:

Take over accounts

Transfer full balances

Rapidly cash out

🛡️ Business Recommendations

Real-time anomaly scoring

Threshold-based fraud blocking (≥ 0.999 probability)

Multi-factor authentication for high-risk transfers

Temporary hold on suspicious transactions

Automated fraud escalation workflow

📈 Measuring Impact

Success should be evaluated using:

Reduction in fraud losses

Monitoring Precision & Recall

False positive trend tracking

A/B testing before full deployment

🏁 Conclusion

This project demonstrates a scalable and production-ready fraud detection framework using behavioral transaction patterns.

The final model achieves strong precision while maintaining acceptable recall, making it suitable for real-world fraud prevention systems.