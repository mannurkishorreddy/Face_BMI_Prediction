import streamlit as st
import pandas as pd
data = pd.read_csv('predictions.csv')

# Set the app title and header
st.title("BMI Prediction App")
st.header("Actual vs Predicted BMI Values")

# Display the data frame
st.dataframe(data)