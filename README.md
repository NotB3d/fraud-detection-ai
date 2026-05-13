# Fraud Detection Project

A machine learning project to detect fraudulent credit card transactions using Random Forest and Logistic Regression classifiers.

## Project Overview

This project includes:
- **Data Processing**: Preprocessing credit card transaction data with StandardScaler
- **Model Training**: 
  - Logistic Regression classifier
  - Random Forest classifier with GridSearchCV hyperparameter tuning
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

### Random Forest (Original)
- **Accuracy**: 99.95%
- **Precision**: 96.05%
- **Recall**: 74.49%

### Random Forest (Tuned with GridSearchCV)
- **Hyperparameters Tuned**: n_estimators, max_depth, min_samples_split
- **Scoring Metric**: Recall (optimized for fraud detection)
- **Improved Recall**: ~83.21% (vs 76.54% original)
- **Best Parameters**: n_estimators=100, max_depth=20, min_samples_split=5

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NotB3d/fraud-detection-ai.git
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

### Train Models with Hyperparameter Tuning
```bash
python hyperparameter_tuning.py
```
This will:
- Load and preprocess the dataset
- Perform GridSearchCV hyperparameter tuning on Random Forest
- Test different values for n_estimators, max_depth, min_samples_split
- Use recall as the primary scoring metric
- Compare original vs tuned model performance
- Generate a comparison bar chart
- Save the tuned model and scaler to `models/` folder

### Original Training (without tuning)
```bash
python main.py
```
This will:
- Load and preprocess the dataset
- Train both classifiers with default parameters
- Save the Random Forest model and scaler to `models/` folder
- Display evaluation metrics

### Run Streamlit App
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

### Generate Comparison Chart Only
```bash
python generate_chart.py
```
This will generate a bar chart comparing recall scores between original and tuned models.

## Hyperparameter Tuning Details

The `hyperparameter_tuning.py` script performs comprehensive hyperparameter optimization:

### Parameters Tested
- **n_estimators**: [50, 100, 200] - Number of trees in the forest
- **max_depth**: [10, 20, None] - Maximum depth of each tree
- **min_samples_split**: [2, 5, 10] - Minimum samples required to split a node

### Optimization Strategy
- **Scoring Metric**: Recall (focuses on detecting fraudulent transactions)
- **Cross-Validation**: 3-fold CV for robust evaluation
- **Search Method**: GridSearchCV for exhaustive parameter search

### Output Files
- `models/fraud_model_tuned.pkl` - Tuned Random Forest model
- `models/scaler_tuned.pkl` - Fitted StandardScaler
- `models/recall_comparison.png` - Performance comparison chart

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
## Autores

Trabajo realizado por:

- Kevin Alexander Román Zaruma  
- Emmanuel Escobar

Asignatura: Artificial Intelligence Foundations  
Institución: Fundació Universitat Rovira i Virgili (URV)