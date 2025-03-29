# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 18:21:43 2025

@author: shaan
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# ✅ Load the trained model
loaded_model = pickle.load(open('C:/Users/shaan/OneDrive/Desktop/trained_model.sav', 'rb'))

# ✅ Load dataset for feature names & fit scaler
diabetes_dataset = pd.read_csv('C:/Users/shaan/OneDrive/Desktop/diabetes.csv')
X = diabetes_dataset.drop(columns='Outcome', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit the scaler on training data

def diabetes_prediction(input_data):
    # ✅ Convert input to DataFrame with column names
    input_data_df = pd.DataFrame([input_data], columns=diabetes_dataset.columns[:-1])

    # ✅ Standardize input data using the trained scaler
    input_data_scaled = scaler.transform(input_data_df)

    # ✅ Make prediction
    prediction = loaded_model.predict(input_data_scaled)

    return "The person is Diabetic" if prediction[0] == 1 else "The person is Not Diabetic"

def main():
    # ✅ Title
    st.title('Diabetes Prediction Web App')

    # ✅ Get user input
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    diagnosis = ''

    # ✅ Prediction button
    if st.button('Diabetes Test Result'):
        try:
            # ✅ Convert input values to float for processing
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                          float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            
            # ✅ Make prediction
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            diagnosis = "⚠ Please enter valid numeric values."

    st.success(diagnosis)

if __name__ == '__main__':
    main()
