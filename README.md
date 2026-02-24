## Proactive Fraud Detection – Case Study Solution

This repository contains my solution to a **fraud detection business case** for a financial company.  
The goal is to build a machine learning model that proactively identifies fraudulent transactions and to derive **actionable business recommendations** from the model’s insights.

The work follows the expectations in the task brief:

- **Data cleaning** (missing values, outliers, multicollinearity)
- **Model design and description**
- **Variable/feature selection process**
- **Model performance demonstration**
- **Key drivers of fraud and interpretation**
- **Infrastructure and process recommendations**
- **Plan to measure impact of the proposed actions**

---

### 1. Project structure

- `project.ipynb` – end‑to‑end notebook with:
  - data loading and exploration
  - data cleaning and preprocessing
  - feature engineering
  - model training and evaluation
  - insights and business recommendations
- `Data Dictionary.txt` – description of the dataset fields.
- `Fraud.csv` – *local* CSV dataset (~6.3M rows, ignored in GitHub due to size limits; should be downloaded separately from the original source).
- `.gitignore` – ignores large/raw data files and notebook artefacts.

To run the notebook you need Python 3.9+, Jupyter, and standard data‑science libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).

---

### 2. Business problem and data

- **Business question**: Predict which financial transactions are fraudulent so that the company can **block or review them in real time**, reducing monetary loss and improving customer trust.
- **Data**:
  - ~6.36M transactions, **11 features**, heavily **imbalanced** with ~0.13% labeled as fraud.
  - Fraud is observed only for the transaction types **`TRANSFER`** and **`CASH_OUT`**.
  - The dataset is provided in CSV format (`Fraud.csv`), and a separate data dictionary explains each field.

---

### 3. Data cleaning and preprocessing

The following steps address the first requirement: **“Data cleaning including missing values, outliers and multi‑collinearity.”**

- **Missing values**
  - Verified that the dataset has **no missing values**, so no imputation was required.

- **Duplicates and data sanity**
  - Checked and removed **duplicate records** if present.
  - Filtered out **irrelevant transaction types** where fraud does not occur, focusing modeling on types with actual fraud (`TRANSFER`, `CASH_OUT`).

- **Leakage and target contamination**
  - Removed **data leakage columns**, i.e. variables that would not be available at prediction time or that directly encode the target:
    - `type` (when encoded in a way that leaks fraud patterns)
    - `isFlaggedFraud`
    - technical / ID columns
  - This ensures the model generalizes to unseen data and mimics a real‑time decisioning scenario.

- **Outliers**
  - Transaction amounts are highly skewed; large legitimate transfers exist alongside fraudulent ones.
  - Instead of blindly trimming, outliers were treated via:
    - log‑scaling / transformations where appropriate
    - using **robust metrics** and models less sensitive to extreme values
  - Outliers are also handled indirectly via **engineered balance‑based features**, where abnormal jumps in balances are more informative than raw amounts alone.

- **Multicollinearity**
  - Investigated correlation between numerical variables and engineered features.
  - Highly collinear variables (e.g. raw balances vs. derived balance differences) were monitored to avoid redundant information.
  - Final model uses a compact set of interpretable features that minimize multicollinearity while preserving predictive power.

- **Class imbalance**
  - Fraudulent transactions are extremely rare (0.13%).
  - Used **`class_weight='balanced'`** in logistic regression so that the model gives more importance to the minority class without manual resampling.

---

### 4. Model description

This section addresses: **“Describe your fraud detection model in elaboration.”**

- **Model choice**: **Logistic Regression** with `class_weight='balanced'`.

- **Why logistic regression**
  - **Interpretable**: coefficients directly indicate direction and strength of each feature’s impact on fraud probability.
  - **Scalable and efficient** on millions of rows.
  - **Good baseline** for production – simple to deploy, monitor, and recalibrate.
  - Works well with **engineered features** that summarize transaction behaviour.

- **Training setup**
  - Split data into **calibration (train)** and **validation (test)** sets.
  - Standardized/normalized features where needed.
  - Used the **training set** to fit the model and **validation set** to evaluate performance and avoid overfitting.

---

### 5. Feature / variable selection

This addresses: **“How did you select variables to be included in the model?”**

Variables were selected using a mix of **domain knowledge** and **data‑driven analysis**:

- **Initial candidate variables**
  - Original transaction features (amounts, source and destination balances, transaction type, etc.).
  - Derived labels from the dataset (`isFraud` as the target).

- **Engineered behavioural features**
  - `orig_balance_diff` – change in the origin account’s balance around the transaction.
  - `dest_balance_diff` – change in the destination account’s balance.
  - `orig_balance_error` – discrepancy between expected and reported origin balance after the transaction.
  - `dest_balance_error` – discrepancy on the destination side.
  - These features capture **abnormal balance movements** that are typical of fraudulent activity (e.g. emptying an account in one go).

- **Selection process**
  - Removed **leakage and ID columns**.
  - Analysed **univariate relationships** between features and fraud label.
  - Monitored **correlation and multicollinearity** to avoid redundant variables.
  - Kept features that are:
    - Predictive (improve ROC‑AUC / F1 score),
    - Stable (behave similarly across train vs. validation),
    - **Business‑interpretable** (can be explained to risk/compliance teams).

---

### 6. Model performance

This addresses: **“Demonstrate the performance of the model by using best set of tools.”**

On the validation set, the final logistic regression model achieved:

- **ROC‑AUC**: **0.9736**
- **Precision** (fraud class): **80%**
- **Recall** (fraud class): **48%**
- **F1‑score**: **0.60**
- **False positives**: **196**

**Confusion matrix**:

|                       | Predicted Not Fraud | Predicted Fraud |
|-----------------------|---------------------|-----------------|
| **Actual Not Fraud**  | 552,243             | 196             |
| **Actual Fraud**      | 862                 | 781             |

Interpretation:

- The model is **highly discriminative** (ROC‑AUC ≈ 0.97).
- With the chosen threshold, **80% of flagged transactions are actually fraud**, which is good for operational efficiency.
- Recall of 48% means roughly **half of fraudulent transactions are caught**; this can be tuned further depending on risk appetite (trading off more false positives for higher recall).

---

### 7. Key fraud drivers and interpretation

These points address:
- **“What are the key factors that predict fraudulent customer?”**
- **“Do these factors make sense? If yes, how?”**

From the model coefficients and feature importance analysis, fraudulent transactions typically exhibit:

- **Sudden full balance depletion**
  - Origin accounts often move **almost their entire balance** in one transaction.
  - Captured by large `orig_balance_diff` and high `orig_balance_error`.

- **Large abnormal balance transitions**
  - Very high transaction amounts relative to normal activity and balances.
  - Significant jumps on the destination side (`dest_balance_diff`, `dest_balance_error`).

- **Mismatch between transaction amount and balance changes**
  - Inconsistent relationships between `amount`, pre‑transaction balance, and post‑transaction balance.
  - These inconsistencies are strong red flags of fraudulent manipulation.

- **Transaction type**
  - Fraud is concentrated in **`TRANSFER`** and **`CASH_OUT`** operations, which are natural channels for account takeovers and rapid cash‑outs.

These factors **make business sense**:

- Fraudsters typically **take over an account**, **transfer out the full balance**, and **cash out quickly**, leaving the account empty.
- The engineered balance features are aligned with classic fraud patterns observed in banking and payments.

---

### 8. Recommended prevention strategies

This answers: **“What kind of prevention should be adopted while company update its infrastructure?”**

Based on the model and insights, the following measures are recommended:

- **Real‑time anomaly scoring**
  - Deploy the model as an online scoring service.
  - Score each `TRANSFER` and `CASH_OUT` transaction in real time.

- **Threshold‑based blocks and reviews**
  - Configure **high‑risk thresholds**, for example:
    - **Auto‑block** transactions with predicted fraud probability ≥ 0.999.
    - **Route to manual review** for probabilities in a “grey zone” (e.g. 0.9–0.999).

- **Stronger authentication for high‑risk events**
  - Trigger **multi‑factor authentication (MFA)** when:
    - large amounts are moved,
    - balance is nearly depleted,
    - device/location is unusual.

- **Temporary transaction holds**
  - Place **short, reversible holds** on suspicious high‑value transfers until customer confirmation is obtained.

- **Automated fraud operations workflow**
  - Integrate alerts with fraud operations tools:
    - case creation,
    - investigator dashboards,
    - feedback loop from confirmed fraud/legit decisions back into the model training data.

---

### 9. Measuring impact and continuous improvement

This answers: **“Assuming these actions have been implemented, how would you determine if they work?”**

To evaluate and continuously improve the system:

- **Key outcome metrics**
  - Reduction in **fraud loss amount** (absolute and as a percentage of transaction volume).
  - **Detection rate (recall)**: proportion of fraudulent transactions blocked or flagged.
  - **Precision / false positive rate**: fraction of blocked transactions that are actually fraud.

- **Monitoring and reporting**
  - Track **precision, recall, F1**, and ROC‑AUC on a rolling window (daily/weekly).
  - Monitor **false positive trends** by segment (customer type, geography, channel).
  - Watch for **model drift** – changes in feature distributions or performance over time.

- **A/B testing and experimentation**
  - Run **A/B tests** comparing:
    - current rule‑based system vs. model‑driven system, or
    - different thresholds and policies.
  - Measure:
    - fraud losses,
    - customer friction (extra MFA, blocked transactions),
    - operational load on investigators.

- **Feedback loop**
  - Use confirmed fraud / non‑fraud decisions to:
    - retrain and recalibrate the model,
    - refine business rules and thresholds,
    - update monitoring dashboards.

---

### 10. How to reproduce the analysis

1. Install Python and dependencies:
   - `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`.
2. Place the dataset as `Fraud.csv` in the project folder (downloaded from the original source; not stored in this GitHub repo due to size constraints).
3. Launch:

```bash
jupyter notebook project.ipynb
```

4. Run cells sequentially to:
   - clean and preprocess data,
   - engineer features,
   - train and evaluate the model,
   - review the insights and recommendations described above.