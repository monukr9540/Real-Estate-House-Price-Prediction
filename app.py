import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model using joblib
model = joblib.load("MK.joblib")

# Streamlit app title
st.title("House Price Prediction App")

# Form to capture user input for house features
st.header("Input House Features")

# Create input fields based on the dataset columns
CRIM = st.number_input("Crime Rate per Capita (CRIM)", value=0.01, key='CRIM')
ZN = st.number_input("Residential Land Zoned (ZN)", value=0.0, key='ZN')
INDUS = st.number_input("Non-Retail Business Acres (INDUS)", value=5.0, key='INDUS')
CHAS = st.selectbox("Bound Charles River (CHAS)", options=[0, 1], key='CHAS')
NOX = st.number_input("Nitric Oxides Concentration (NOX)", value=0.5, key='NOX')
RM = st.number_input("Average Number of Rooms (RM)", value=6.0, key='RM')
AGE = st.number_input("Proportion of Older Homes (AGE)", value=50.0, key='AGE')
DIS = st.number_input("Distances to Employment Centers (DIS)", value=4.0, key='DIS')
RAD = st.number_input("Accessibility to Highways (RAD)", value=5, key='RAD')
TAX = st.number_input("Property Tax Rate (TAX)", value=300, key='TAX')
PTRATIO = st.number_input("Pupil-Teacher Ratio (PTRATIO)", value=15.0, key='PTRATIO')
B = st.number_input("Proportion of Black Residents (B)", value=390.0, key='B')
LSTAT = st.number_input("Lower Status of Population (LSTAT)", value=10.0, key='LSTAT')

# Button to predict
if st.button("Predict House Price"):
    # Collecting the input into a numpy array
    user_input = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    
    # Make predictions (assuming the model is already trained with scaled data or no scaling is required)
    prediction = model.predict(user_input)
    
    # Display prediction
    st.subheader(f"Predicted House Price: ${prediction[0]:,.2f}")

# You can also add a sidebar or data visualization if needed
