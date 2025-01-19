import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load your trained model (ensure the path is correct)
GBC = joblib.load('heart_attack_model.pkl')  # Replace with your model file path

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 50)
    heart_rate = st.sidebar.slider('Heart Rate', 40, 180, 70)
    diabetes = st.sidebar.selectbox('Diabetes (0 for No, 1 for Yes)', [0, 1])
    family_history = st.sidebar.selectbox('Family History (0 for No, 1 for Yes)', [0, 1])
    smoking = st.sidebar.selectbox('Smoking (0 for No, 1 for Yes)', [0, 1])
    obesity = st.sidebar.selectbox('Obesity (0 for No, 1 for Yes)', [0, 1])
    alcohol_consumption = st.sidebar.selectbox('Alcohol Consumption (0 for No, 1 for Yes)', [0, 1])
    exercise_hours_per_week = st.sidebar.slider('Exercise Hours Per Week', 0, 20, 3)
    previous_heart_problems = st.sidebar.selectbox('Previous Heart Problems (0 for No, 1 for Yes)', [0, 1])
    stress_level = st.sidebar.slider('Stress Level', 0, 10, 5)
    sedentary_hours_per_day = st.sidebar.slider('Sedentary Hours Per Day', 0, 12, 5)
    bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
    triglycerides = st.sidebar.slider('Triglycerides', 50, 400, 150)
    sleep_hours_per_day = st.sidebar.slider('Sleep Hours Per Day', 4, 12, 7)
    systolic = st.sidebar.slider('Systolic BP', 90, 180, 120)
    diastolic = st.sidebar.slider('Diastolic BP', 60, 120, 80)

    # Checkbox inputs for categorical data
    sex_female = st.sidebar.checkbox('Female')
    sex_male = st.sidebar.checkbox('Male')
    diet_average = st.sidebar.checkbox('Average Diet')
    diet_healthy = st.sidebar.checkbox('Healthy Diet')
    diet_unhealthy = st.sidebar.checkbox('Unhealthy Diet')
    cholesterol_normal = st.sidebar.checkbox('Normal Cholesterol')
    cholesterol_at_risk = st.sidebar.checkbox('Cholesterol At Risk')
    cholesterol_high = st.sidebar.checkbox('High Cholesterol')
    cholesterol_dangerous = st.sidebar.checkbox('Dangerous Cholesterol')

    data = {
        'age': Age,
        'heart_rate': Heart Rate,
        'diabetes': Diabetes,
        'family_history': Family History,
        'smoking': Smoking,
        'obesity': Obesity,
        'alcohol_consumption': Alcohol Consumption,
        'exercise_hours_per_week': Exercise Hours Per Week,
        'previous_heart_problems': Previous Heart Problems,
        'stress_level': Stress Level,
        'sedentary_hours_per_day': Sedentary Hours Per Day,
        'bmi': BMI,
        'triglycerides': Triglycerides,
        'sleep_hours_per_day': Sleep Hours Per Day,
        'systolic': Systolic,
        'diastolic': Diastolic,
        'sex_female': 1 if Sex_Female else 0,
        'sex_male': 1 if Sex_Male else 0,
        'diet_average': 1 if Diet_Average else 0,
        'diet_healthy': 1 if Diet_Healthy else 0,
        'diet_unhealthy': 1 if Diet_Unhealthy else 0,
        'cholesterol_normal': 1 if Cholesterol_Normal else 0,
        'cholesterol_at_risk': 1 if Cholesterol_At Risk else 0,
        'cholesterol_high': 1 if Cholesterol_High else 0,
        'cholesterol_dangerous': 1 if Cholesterol_Dangerous else 0
    }

    return pd.DataFrame(data, index=[0])

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# Make prediction
prediction = GBC.predict(df)
prediction_proba = GBC.predict_proba(df)

st.subheader('Prediction')
st.write('Heart Attack Risk:', 'High' if prediction[0] == 1 else 'Low')

st.subheader('Prediction Probability')
st.write(f"Probability of High Risk: {prediction_proba[0][1]:.2f}")
st.write(f"Probability of Low Risk: {prediction_proba[0][0]:.2f}")
