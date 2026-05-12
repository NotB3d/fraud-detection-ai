import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load the credit card fraud dataset (assuming 'creditcard.csv' is in the current directory)
df = pd.read_csv('creditcard.csv')

# Preprocess Time and Amount using StandardScaler
scaler = StandardScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

# Prepare features and labels for training
y = df['Class']
X = df.drop(columns=['Class'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a Logistic Regression model
model_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model_lr.fit(X_train, y_train)

# Predict and evaluate on the test set
y_pred_lr = model_lr.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)

# Train a Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model_rf.fit(X_train, y_train)

# Predict and evaluate on the test set
y_pred_rf = model_rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Create models folder if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the Random Forest model and scaler
joblib.dump(model_rf, 'models/fraud_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Random Forest model saved to models/fraud_model.pkl")
print("Scaler saved to models/scaler.pkl")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution (fraud vs non-fraud):")
print(df['Class'].value_counts())

print("\nLogistic Regression evaluation on the test set:")
print(f"Accuracy: {accuracy_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall: {recall_lr:.4f}")
print("Confusion matrix:")
print(conf_matrix_lr)

print("\nRandom Forest evaluation on the test set:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print("Confusion matrix:")
print(conf_matrix_rf)
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.savefig('class_distribution.png')
plt.close()
