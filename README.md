<<<<<<< HEAD
# Fraud Detection Project

A machine learning project to detect fraudulent credit card transactions using Random Forest and Logistic Regression classifiers.

## Project Overview

This project includes:
- **Data Processing**: Preprocessing credit card transaction data with StandardScaler
- **Model Training**: 
  - Logistic Regression classifier
  - Random Forest classifier (100 estimators)
- **Model Comparison**: Accuracy, Precision, Recall, and Confusion Matrix metrics
- **Streamlit Web App**: Interactive interface for single and batch fraud predictions

## Dataset

Uses the Credit Card Fraud Detection dataset with 284,807 transactions and 31 features:
- `Time`: Seconds elapsed between transactions
- `Amount`: Transaction amount in dollars
- `V1-V28`: Anonymized PCA-transformed features
- `Class`: Target variable (0 = Legitimate, 1 = Fraudulent)

## Model Performance

### Logistic Regression
- **Accuracy**: 97.55%
- **Precision**: 6.09%
- **Recall**: 91.84%

### Random Forest (Best Model)
- **Accuracy**: 99.95%
- **Precision**: 96.05%
- **Recall**: 74.49%

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd fraud-detection-project
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train Models
```bash
python main.py
```
This will:
- Load and preprocess the dataset
- Train both classifiers
- Save the Random Forest model and scaler to `models/` folder
- Display evaluation metrics

### Run Streamlit App
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

The app provides:
- **Manual Input**: Predict fraud for a single transaction
- **Batch Prediction**: Upload CSV file to predict multiple transactions

## Project Structure

```
fraud-detection-project/
├── main.py                 # Model training script
├── app.py                  # Streamlit web application
├── creditcard.csv          # Dataset (not included in repo)
├── models/
│   ├── fraud_model.pkl     # Trained Random Forest model
│   └── scaler.pkl          # StandardScaler for preprocessing
├── class_distribution.png  # Class distribution visualization
├── requirements.txt        # Python dependencies
└── .gitignore             # Git ignore file
```

## Dependencies

- pandas
- scikit-learn
- matplotlib
- seaborn
- streamlit
- joblib

## Features

- **Feature Engineering**: Time and Amount standardization
- **Class Balancing**: `class_weight='balanced'` to handle imbalanced data
- **Train-Test Split**: Stratified 80-20 split
- **Model Persistence**: Save/load trained models with joblib
- **Web Interface**: User-friendly Streamlit app for predictions

## Contributing

Feel free to fork, create issues, and submit pull requests!

## License

MIT License - feel free to use this project for your own purposes.
=======
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
>>>>>>> 126b9318397bd2b402832a86244aeca71a5e854a
