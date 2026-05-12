# fraud-detection-ai
Machine Learning project for credit card fraud detection using Logistic Regression and Random Forest with a Streamlit interactive dashboard.

# 💳 Credit Card Fraud Detection using Machine Learning

Project Overview
This project implements a machine learning system to detect fraudulent credit card transactions using real anonymized transaction data. The system compares different classification models and provides an interactive Streamlit application for real-time prediction.

---

 Objective
The goal is to build a supervised learning model capable of distinguishing between legitimate and fraudulent transactions while handling a highly imbalanced dataset.

---

Dataset
- Source: Kaggle Credit Card Fraud Dataset
- Total transactions: 284,807
- Features: 30 (Time, Amount, V1–V28)
- Target variable:
  - 0 → Legitimate transaction
  - 1 → Fraudulent transaction

 Note: V1–V28 are PCA-transformed features (anonymized for privacy).

---

 Technologies Used
- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Streamlit
- Joblib

---

Machine Learning Models
- Logistic Regression
- Random Forest Classifier

Both models were trained using `class_weight='balanced'` due to class imbalance.

---

 Results

| Model | Accuracy | Precision | Recall |
|------|----------|----------|--------|
| Logistic Regression | 97.55% | 6.09% | 91.84% |
| Random Forest | 99.95% | 96.05% | 74.49% |

---

 Key Insights
- Dataset is highly imbalanced (fraud << normal transactions)
- Logistic Regression achieves high recall (detects more frauds)
- Random Forest achieves higher precision (fewer false positives)
- Trade-off between recall and precision is critical in fraud detection

---

Streamlit Application

The interactive app allows:

- Loading real fraud or legitimate transactions
- Viewing anonymized features (V1–V28)
- Predicting fraud in real time
- Comparing predictions with actual labels

### ▶️ Run the app locally:
git clone https://github.com/NotB3d/fraud-detection-ai.git
cd fraud-detection-ai |
pip install -r requirements.txt |
streamlit run src/app.py 
