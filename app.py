import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Load the saved model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('models/fraud_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Title and description
st.title("💳 Credit Card Fraud Detection System")
st.markdown("""
This app uses a trained Random Forest classifier to predict whether a credit card transaction is fraudulent or legitimate.
""")

# Define the expected feature order from training
FEATURE_ORDER = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Manual Input", "Batch Prediction"])

with tab1:
    st.header("Single Transaction Prediction")
    
    col1, col2 = st.columns(2)
    
    # Time and Amount inputs
    with col1:
        st.subheader("Transaction Details")
        time_input = st.number_input("Time (seconds since first transaction)", value=0, min_value=0)
        amount_input = st.number_input("Transaction Amount ($)", value=100.0, min_value=0.0)
    
    # PCA features (V1-V28)
    with col2:
        st.subheader("PCA Features")
        st.info("Enter values for all 28 PCA features (V1-V28). Typical range is -3 to +3.")
    
    # Create input fields for V1-V28
    features_dict = {'Time': time_input, 'Amount': amount_input}
    
    cols = st.columns(4)
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        with cols[col_idx]:
            features_dict[f'V{i}'] = st.number_input(f"V{i}", value=0.0, step=0.1, key=f"v{i}")
    
    # Make prediction
    if st.button("🔍 Predict Transaction", key="predict_single", use_container_width=True):
        # Prepare data
        data = pd.DataFrame([features_dict])
        
        # Scale Time and Amount
        data[['Time', 'Amount']] = scaler.transform(data[['Time', 'Amount']])
        
        # Reorder columns to match the training feature order
        data = data[FEATURE_ORDER]
        
        # Make prediction
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0]
        
        # Display results
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.success("✅ **Prediction: LEGITIMATE**")
                st.metric("Confidence", f"{probability[0]*100:.2f}%")
            else:
                st.error("⚠️ **Prediction: FRAUDULENT**")
                st.metric("Confidence", f"{probability[1]*100:.2f}%")
        
        with col2:
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame({
                'Class': ['Legitimate', 'Fraudulent'],
                'Probability': [probability[0], probability[1]]
            })
            st.bar_chart(prob_df.set_index('Class'))

with tab2:
    st.header("Batch Prediction")
    st.markdown("""
    Upload a CSV file with transaction data to make predictions on multiple transactions at once.
    The CSV should contain all 30 features: Time, Amount, V1-V28.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_upload")
    
    if uploaded_file is not None:
        # Read the CSV
        df = pd.read_csv(uploaded_file)
        
        # Check if all required features are present
        required_features = FEATURE_ORDER.copy()
        missing_features = [f for f in required_features if f not in df.columns]
        
        if missing_features:
            st.error(f"Missing features: {', '.join(missing_features)}")
        else:
            # Select and reorder features exactly as in training
            df_model = df[required_features].copy()
            
            # Scale Time and Amount
            df_model[['Time', 'Amount']] = scaler.transform(df_model[['Time', 'Amount']])
            
            # Make predictions
            predictions = model.predict(df_model)
            probabilities = model.predict_proba(df_model)
            
            # Add predictions to dataframe
            df['Prediction'] = ['Fraudulent' if p == 1 else 'Legitimate' for p in predictions]
            df['Fraud_Probability'] = probabilities[:, 1]
            df['Legitimate_Probability'] = probabilities[:, 0]
            
            # Display results
            st.subheader("Predictions")
            st.dataframe(df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", len(df))
            with col2:
                fraud_count = (predictions == 1).sum()
                st.metric("Fraudulent Transactions", fraud_count)
            with col3:
                st.metric("Fraud Rate", f"{fraud_count/len(df)*100:.2f}%")
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

# Footer
st.divider()
st.markdown("""
---
**Model Information:**
- Algorithm: Random Forest Classifier (100 estimators)
- Training: Credit Card Fraud Detection Dataset
- Features: 30 (Time, Amount, V1-V28)
""")
