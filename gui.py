import streamlit as st
import joblib
import numpy as np

# Set up the Streamlit app
st.title("Salary Estimation App")
st.divider()

# Input fields
years_at_company = st.number_input("Enter years at company", min_value=0, max_value=20)
satisfaction_level = st.number_input("Satisfaction Level", min_value=0.0)
average_monthly_hours = st.number_input("Average Monthly Hours", min_value=120, max_value=400)

# Prepare input data
X = [years_at_company, satisfaction_level, average_monthly_hours]

# Load the scaler
scaler = joblib.load(r"D:\Code\Project\scaler.pkl")

# Load the models
svm_model = joblib.load(r"D:\Code\Project\svm_model.pkl")
linear_model = joblib.load(r"D:\Code\Project\linear_regression_model.pkl")

# Model selection dropdown
model_option = st.selectbox("Choose a model", ["Support Vector Machine (SVM)", "Linear Regression"])

# Predict button
predict_button = st.button("Press for predicting the salary")
st.divider()

if predict_button:
    st.balloons()
    X1 = np.array(X)  # Convert input to numpy array
    X_array = scaler.transform([X1])  # Scale the input

    # Select the model based on user choice
    if model_option == "Support Vector Machine (SVM)":
        model = svm_model
    else:
        model = linear_model

    # Make prediction
    prediction = np.array(model.predict(X_array), ndmin=2)  # Convert to 2D array
    rounded_prediction = round(prediction[0][0])  # Round to nearest integer
    formatted_prediction = f"${rounded_prediction:,}"  # Format as currency
    st.write(f"Salary prediction is {formatted_prediction}")
else:
    st.write("Please enter the values and press the predict button")