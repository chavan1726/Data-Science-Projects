import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np

# Set page title and layout
st.set_page_config(page_title="Customer Spending Predictor", layout="wide")

# Custom CSS for background image
st.markdown(
    """
    <style>
    .main {
        background-image: url('https://th.bing.com/th/id/OIP.6rHjdwhwrL_VCpWWTh1m_gHaHa?pid=ImgDet&w=172&h=172&c=7&dpr=1.1');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the data
data = pd.read_csv("Ecommerce_Customers (2).csv")

# Data preparation
X = data[['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = data['Yearly Amount Spent']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Main title
st.title("Customer Spending Prediction")
st.markdown("""
This application predicts the *Yearly Amount Spent* by customers based on their behavior metrics. 
Please use the sidebar to enter customer details for prediction.
""")

# Sidebar inputs
st.sidebar.header("Customer Input")
avg_session_length = st.sidebar.number_input("Average Session Length", min_value=0.0, step=0.1)
time_on_app = st.sidebar.number_input("Time on App", min_value=0.0, step=0.1)
time_on_website = st.sidebar.number_input("Time on Website", min_value=0.0, step=0.1)
length_of_membership = st.sidebar.number_input("Length of Membership", min_value=0.0, step=0.1)

# Model performance metrics
st.subheader("Model Performance")
col1, col2, col3,col4= st.columns(4)
with col1:
    st.metric(label="Mean Absolute Error", value=f"{mean_absolute_error(y_test, y_pred):.2f}")
with col2:
    st.metric(label="Mean Squared Error", value=f"{mean_squared_error(y_test, y_pred):.2f}")
with col3:
    st.metric(label="Root Mean Squared Error", value=f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
with col4:
    st.metric(label="R2_Score", value=f"{r2_score(y_test, y_pred):.2f}")

# Prediction
if st.sidebar.button("Predict"):
    new_data = [[avg_session_length, time_on_app, time_on_website, length_of_membership]]
    prediction = model.predict(new_data)
    st.subheader("Prediction Result")
    st.success(f"Predicted Yearly Amount Spent: *${prediction[0]:.2f}*")

