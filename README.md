Diabetes Detection System

Overview

This is a Diabetes Detection System built using Streamlit and Machine Learning. The application allows users to input medical parameters, and based on these inputs, it predicts whether a person has diabetes using a trained Support Vector Machine (SVM) model.

Features

User-friendly web interface using Streamlit

Input fields for necessary medical parameters

Machine learning model for diabetes prediction

Real-time prediction results

Uses trained_model.sav for inference

Installation

Ensure you have Python 3.7+ installed on your system.

Step 1: Clone the Repository

git clone https://github.com/Shaan-Thakkar-1812/Diabetes_Detection.git
cd Diabetes_Detection

Step 2: Install Dependencies

Manually install the required dependencies:

pip install streamlit pandas numpy scikit-learn joblib

Running the Application

streamlit run diabetes_prediction_webapp.py

Project Structure

├── Diabetes_Detection
│   ├── diabetes_prediction_webapp.py   # Main Streamlit application
│   ├── predictive_system.py            # Model prediction logic
│   ├── trained_model.sav               # Pre-trained SVM model
│   ├── diabetes.csv                     # Dataset used for training
│   ├── SAPproject.ipynb                 # Jupyter Notebook with model training steps
│   ├── README.md                        # Documentation

How the Model Works

The Support Vector Machine (SVM) model was trained using the diabetes.csv dataset.

The trained model is saved as trained_model.sav using joblib.

The Streamlit app (diabetes_prediction_webapp.py) loads this model to make real-time predictions.

Users enter relevant medical details in the web interface, and the model predicts if they are diabetic or not.

Notes

If the application does not open automatically, copy the local URL displayed in the terminal and open it in a browser.

Ensure that trained_model.sav exists in the project directory before running the application.
