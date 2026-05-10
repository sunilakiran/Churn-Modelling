# 🏦 Customer Churn Prediction — Bank Customers

A machine learning project to identify bank customers who are likely to leave, using the Churn Modelling dataset with 10,000 customer records.

---

## 📋 Project Overview

Customer churn is one of the most costly problems for banks. Losing a customer means losing their lifetime value. This project builds a classification model that predicts which customers are at risk of leaving, so the bank can take proactive action before it's too late.

---

## 🗂️ Dataset

**File:** `Churn_Modelling.csv`

**Source:** [Kaggle — Churn Modelling Dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling)

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Country — France, Germany, Spain |
| Gender | Male / Female |
| Age | Customer's age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Has credit card? (0/1) |
| IsActiveMember | Active bank member? (0/1) |
| EstimatedSalary | Estimated annual salary |
| **Exited** | **Target — Did the customer leave? (0/1)** |

---

## 🛠️ Tools & Libraries

```
Python 3
├── pandas          — data loading and manipulation
├── numpy           — numerical operations
├── matplotlib      — plotting charts
├── seaborn         — heatmaps and styled plots
└── scikit-learn    — encoding, scaling, models, evaluation
```

---

## ⚙️ Project Pipeline

### Step 1 — Data Cleaning
- Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`
- Checked for missing values — none found
- Dataset shape after cleaning: **(10,000 rows × 11 columns)**

### Step 2 — Encoding Categorical Features

**Label Encoding** → used for `Gender` (binary column)
```
Female → 0
Male   → 1
```

**One-Hot Encoding** → used for `Geography` (3 categories)
```
Geography → Geography_France | Geography_Germany | Geography_Spain
```
One-Hot Encoding was chosen for Geography to avoid implying any ordinal relationship between countries.

### Step 3 — Train / Test Split
- Split ratio: **80% training / 20% testing**
- Stratified split to preserve the churn ratio in both sets
- `StandardScaler` applied for Logistic Regression only
- `class_weight='balanced'` used to handle class imbalance

### Step 4 — Model Training
Three classification models were trained and compared:

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline, balanced class weights |
| Random Forest | 200 trees, max_depth=10, balanced weights |
| Gradient Boosting | 200 trees, learning_rate=0.05, max_depth=5 |

### Step 5 — Evaluation & Feature Importance
Models evaluated using Accuracy, AUC-ROC, F1-score, Precision, and Recall.
Feature importance extracted from Random Forest's `feature_importances_`.

---

## 📊 Results

### Model Comparison

| Model | Accuracy | AUC-ROC | F1-Score |
|---|---|---|---|
| Logistic Regression | 71.35% | 77.71% | 49.87% |
| Random Forest | 83.95% | 86.16% | 61.92% |
| **Gradient Boosting** | **86.80%** | **86.54%** | **60.24%** |

### Classification Report Highlights (Churned Class)

| Model | Precision | Recall | Meaning |
|---|---|---|---|
| Logistic Regression | 39% | 70% | Catches most churners but many false alarms |
| Random Forest | 60% | 64% | Best balance — fewer false alerts |
| Gradient Boosting | 78% | 49% | Precise but misses half of real churners |

---

## 🏆 Best Model — Random Forest

Although Gradient Boosting achieved the highest accuracy (86.8%), it only detected **49% of actual churners** — meaning it would miss every second at-risk customer.

**Random Forest** was selected as the best model because:
- Strong AUC-ROC of **86.16%** — excellent at ranking churn risk
- Recall of **64%** — catches most real churners
- Precision of **60%** — fewer wasted retention efforts
- Best overall balance between catching churners and avoiding false alarms

> **Key insight:** For a bank, missing a churner is more costly than a false alarm. Recall matters more than raw accuracy.

---

## 📌 Feature Importance (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Age | **31.5%** |
| 2 | NumOfProducts | **20.1%** |
| 3 | Balance | **12.0%** |
| 4 | EstimatedSalary | 8.3% |
| 5 | CreditScore | 8.2% |
| 6 | IsActiveMember | 5.2% |
| 7 | Tenure | 4.7% |
| 8 | Geography_Germany | 4.3% |
| 9 | Gender | 2.2% |
| 10 | Geography_France | 1.4% |
| 11 | HasCrCard | 1.1% |
| 12 | Geography_Spain | 1.0% |

Only **Age**, **NumOfProducts**, and **Balance** crossed the 10% importance threshold.

---

## 💡 Key Business Insights

**1. Age is the #1 churn driver (31.5%)**
Older customers are significantly more likely to leave. The bank should build dedicated retention programs for customers aged 50 and above.

**2. Number of products is counterintuitive (20.1%)**
Customers with 3 or 4 products churn more despite appearing highly engaged. This may signal product dissatisfaction or being over-sold.

**3. High balance customers are at risk (12.0%)**
Wealthy customers with large balances are more likely to churn — possibly because they are actively seeking better rates or premium services elsewhere.

**4. Inactive members churn silently (5.2%)**
Customers marked as inactive are approximately 2× more likely to churn. Re-engagement campaigns should target this group early.

**5. Germany branch has higher churn (4.3%)**
German customers churn at a noticeably higher rate compared to France and Spain, warranting a regional investigation.

---

## 📁 Project Structure

```
bank-churn-prediction/
│
├── Churn_Modelling.csv              ← Raw dataset
├── bank_churn_prediction_colab.py   ← Full Colab code (17 cells)
├── README.md                        ← This file
│
└── outputs/
    ├── churn_distribution.png       ← Churn count & pie chart
    ├── churn_categorical.png        ← Churn by Geography, Gender, Products
    ├── churn_numerical.png          ← Feature distributions by churn status
    ├── correlation_heatmap.png      ← Feature correlation heatmap
    ├── confusion_matrices.png       ← All 3 model confusion matrices
    ├── roc_curves.png               ← ROC curves comparison
    └── feature_importance.png       ← Random Forest feature importance
```

---

## 🚀 How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Copy and paste cells from `bank_churn_prediction_colab.py` one by one
4. Run **Cell 1** first (imports), then upload the dataset in **Cell 2**
5. Run all remaining cells in order

---

