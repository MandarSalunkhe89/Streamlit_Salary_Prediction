import streamlit as st
import joblib
from sklearn import linear_model
import pandas as pd

st.title("Salary Prediction")
st.subheader("Salary prediction based on candidate details")

# User inputs
name = st.text_input('First name')
surname = st.text_input('Last name')
experience = st.number_input("Enter the years of experience here")
skills = st.number_input("Enter the skillset OR techstack")

# import DataFrame
user_input = pd.DataFrame(dict(x1_Experience=[experience], X2_Skills=[skills]))

# import model Pickle
reg = joblib.load('G:\Mandar\__Pycharm_st_ML\Model\predict_salary.pkl')

# ML Model
predicted_salary = reg.predict(user_input)[0]

# Display the predicted output
st.write(name + ' ' + surname + ' ' + 'predicted_salary')
st.write(predicted_salary)