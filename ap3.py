import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# Title and Description
st.title("Urban Land Cover Prediction (Regression)")
st.markdown("""
This application predicts urban land cover using various ML models.
Select a model, provide inputs for key features, and explore predictions and performance metrics.
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

# Sidebar for model selection
st.sidebar.header("Model Selection")
st.sidebar.markdown("""
This implementation is based on the research paper:

**[Predicting Urban Land Cover Using Classification: A Machine Learning Approach](https://ieeexplore.ieee.org/document/10461930/metrics#metrics)**, IEEE.

Explore various models for predicting urban land cover.
""")

model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "KNN", "SVM", "KMeans"])

# Initialize the chosen model
model = None
if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
elif model_choice == "KNN":
    model = KNeighborsRegressor(n_neighbors=5)
elif model_choice == "SVM":
    model = SVR(kernel="linear")
elif model_choice == "KMeans":
    model = KMeans(n_clusters=3, random_state=42)

# Train and evaluate the model (if applicable)
if model is not None:
    if model_choice != "KMeans":  # For supervised models
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        st.header("Model Performance")
        st.write(f"**Train R² Score:** {train_r2:.2f}")
        st.write(f"**Test R² Score:** {test_r2:.2f}")
        
    else:  # For unsupervised models
        model.fit(X_train)
        st.write("**KMeans Clustering Labels:**")
        st.write(model.labels_)

# Feature Importance or Top Features
if model_choice == "Random Forest":
    feature_importances = model.feature_importances_
    top_features = np.argsort(feature_importances)[::-1][:10]
else:
    top_features = list(range(10))  # Select the first 10 features for other models

# User Input for Prediction
st.header("Make a Prediction")
st.write("Provide values for the selected key features:")

# Select top 10 features dynamically
selected_features = X_train.columns[top_features]  # Dynamically select top 10 features
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
    if model_choice != "KMeans":
        prediction = model.predict(input_df)[0]
        st.write(f"**Predicted Value:** {prediction:.2f}")

        # Highlight predicted value in plot
        st.subheader("Predicted vs Actual Values")
        fig = go.Figure()

        # Scatter plot for actual vs predicted values
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred_test, mode='markers', name='Test Data',
            marker=dict(color='cyan', size=8)
        ))

        # Highlight predicted value
        fig.add_trace(go.Scatter(
            x=[prediction], y=[prediction], mode='markers+text', name='Prediction',
            marker=dict(color='red', size=12),
            text=[f"Predicted: {prediction:.2f}"], textposition="top center"
        ))

        # Add ideal line
        fig.add_trace(go.Scatter(
            x=y_test, y=y_test, mode='lines', name='Ideal', line=dict(color='green', dash='dash')
        ))

        fig.update_layout(
            title="Predicted vs Actual",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            template="plotly_dark",
            legend=dict(x=0, y=1)
        )
        st.plotly_chart(fig)
    else:
        cluster = model.predict(input_df)[0]
        st.write(f"**Predicted Cluster:** {cluster}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: small;">
        Designed and maintained by <b>Kshitiz Garg</b> | Roll No: 2K22/EE/152 <br>
        Co-designed with <b>Dhruv Singla</b> | Roll No: 2K22/EE/98 <br>
        Delhi Technological University <br>
        <a href="https://www.linkedin.com/in/kshitiz-garg-898403207/" target="_blank" style="color: cyan;">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)
