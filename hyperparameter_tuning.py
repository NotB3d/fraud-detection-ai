"""
Hyperparameter Tuning for Random Forest Fraud Detection Model

This script demonstrates hyperparameter tuning using GridSearchCV for a Random Forest
classifier on the credit card fraud detection dataset. The tuning focuses on recall
as the primary metric since fraud detection requires maximizing the detection of
fraudulent transactions.

Tested Parameters:
- n_estimators: [50, 100, 200]
- max_depth: [10, 20, None]
- min_samples_split: [2, 5, 10]

Usage:
python hyperparameter_tuning.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(sample_size=None):
    """Load and preprocess the credit card fraud dataset."""
    print("Loading dataset...")
    if sample_size:
        df = pd.read_csv('creditcard.csv', nrows=sample_size)
    else:
        df = pd.read_csv('creditcard.csv')

    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {y.sum()}")
    print(f"Legitimate cases: {len(y) - y.sum()}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale Time and Amount features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def perform_grid_search(X_train, y_train):
    """Perform GridSearchCV with recall as scoring metric."""
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
    print("="*50)

    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    # Create base Random Forest model
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Perform GridSearchCV with recall as scoring metric
    print("Performing GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=3,
        scoring='recall',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Get best parameters and score
    best_params = grid_search.best_params_
    best_recall = grid_search.best_score_

    print(f"\nBest Parameters: {best_params}")
    print(f"Best Cross-Validation Recall Score: {best_recall:.4f}")

    return best_params, best_recall, grid_search

def train_and_compare_models(X_train, X_test, y_train, y_test, best_params):
    """Train tuned and original models and compare their performance."""
    # Train the tuned model with best parameters
    print("\nTraining tuned Random Forest model...")
    rf_tuned = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    rf_tuned.fit(X_train, y_train)

    # Evaluate tuned model on test set
    y_pred_tuned = rf_tuned.predict(X_test)
    recall_tuned = recall_score(y_test, y_pred_tuned)

    print(f"Tuned Model Test Recall: {recall_tuned:.4f}")

    # Train original model for comparison (default parameters)
    print("\nTraining original Random Forest model...")
    rf_original = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_original.fit(X_train, y_train)

    y_pred_original = rf_original.predict(X_test)
    recall_original = recall_score(y_test, y_pred_original)

    print(f"Original Model Test Recall: {recall_original:.4f}")

    return rf_original, rf_tuned, recall_original, recall_tuned, y_pred_original, y_pred_tuned

def generate_comparison_chart(recall_original, recall_tuned):
    """Generate and save a bar chart comparing recall scores."""
    print("\nGenerating comparison chart...")
    recall_scores = [recall_original, recall_tuned]
    model_names = ['Original RF', 'Tuned RF']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(model_names, recall_scores, color=['skyblue', 'lightgreen'])
    plt.title('Recall Score Comparison: Original vs Tuned Random Forest')
    plt.ylabel('Recall Score')
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar, score in zip(bars, recall_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('models/recall_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Chart saved as 'models/recall_comparison.png'")

def print_comparison_results(recall_original, recall_tuned, y_test, y_pred_original, y_pred_tuned):
    """Print detailed comparison results."""
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"Original Random Forest Recall: {recall_original:.4f}")
    print(f"Tuned Random Forest Recall: {recall_tuned:.4f}")
    print(f"Improvement: {(recall_tuned - recall_original):.4f}")

    # Print detailed classification reports
    print("\n" + "="*50)
    print("CLASSIFICATION REPORTS")
    print("="*50)

    print("\nOriginal Random Forest:")
    print(classification_report(y_test, y_pred_original))

    print("\nTuned Random Forest:")
    print(classification_report(y_test, y_pred_tuned))

def save_models(rf_tuned, scaler):
    """Save the tuned model and scaler."""
    print("\nSaving tuned model and scaler...")
    joblib.dump(rf_tuned, 'models/fraud_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    print("Tuned model saved as 'models/fraud_model.pkl' (overwriting original)")
    print("Scaler saved as 'models/scaler.pkl' (overwriting original)")

def main():
    """Main function to run the hyperparameter tuning pipeline."""
    # Load and preprocess data (use sample_size for faster execution during development)
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(sample_size=50000)

    # Perform GridSearchCV
    best_params, best_recall, grid_search = perform_grid_search(X_train, y_train)

    # Train and compare models
    rf_original, rf_tuned, recall_original, recall_tuned, y_pred_original, y_pred_tuned = train_and_compare_models(
        X_train, X_test, y_train, y_test, best_params
    )

    # Generate comparison chart
    generate_comparison_chart(recall_original, recall_tuned)

    # Print comparison results
    print_comparison_results(recall_original, recall_tuned, y_test, y_pred_original, y_pred_tuned)

    # Save models
    save_models(rf_tuned, scaler)

    print("\nHyperparameter tuning completed successfully!")

if __name__ == "__main__":
    main()