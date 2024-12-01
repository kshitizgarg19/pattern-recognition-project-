import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Title and Description
st.title("Urban Land Cover Prediction (Regression)")
st.markdown("""
This application predicts urban land cover using a Random Forest Regressor. 
The data is preprocessed automatically, and the model's performance is evaluated.
""")

# Load Data
train_data = pd.read_csv("training.csv")
test_data = pd.read_csv("testing.csv")

# Preprocess Data
st.header("Data Preprocessing")
st.write("Automatically converting non-numeric columns to numeric values...")

# Convert non-numeric data to numeric using LabelEncoder
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

# Predictions and Evaluation
st.header("Model Evaluation")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Visualization of Predictions
st.subheader("Predicted vs Actual Values")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Predicted vs Actual")
st.pyplot(fig)
