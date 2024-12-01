import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Title and Description
st.title("Urban Land Cover Prediction (Regression)")
st.markdown("""
This application predicts urban land cover using a Random Forest Regressor. 
Provide values for key features to make predictions.
""")

# Load Data
train_data = pd.read_csv("training.csv")
test_data = pd.read_csv("testing.csv")

# Preprocess Data
st.header("Data Preprocessing")
st.write("Automatically converting non-numeric columns to numeric values...")

# Convert non-numeric data to numeric using factorize
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        train_data[column] = pd.factorize(train_data[column])[0]
        test_data[column] = pd.factorize(test_data[column])[0]

st.success("Data preprocessing complete!")

# Split features and labels
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Train Random Forest Regressor
st.header("Model Training")
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)
st.success("Model training complete!")

# Model Evaluation
st.header("Model Evaluation")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Take Limited User Inputs for Prediction
st.header("Make a Prediction")
st.write("Provide values for the selected key features:")

# Select key features for input
selected_features = list(X_train.columns[:12])  # Select first 12 features
user_input = {}

# Generate input fields for selected features
for col in selected_features:
    user_input[col] = st.number_input(f"Enter {col}:", value=float(X_train[col].mean()))

# Fill in missing features with their training means
input_data = {col: X_train[col].mean() for col in X_train.columns}  # Default values
input_data.update(user_input)  # Update with user inputs

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Make Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"**Predicted Value:** {prediction[0]:.2f}")
