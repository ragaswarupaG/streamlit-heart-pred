# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import GradientBoostingClassifier
# import joblib

# # Load your trained model (ensure the path is correct)
# GBC = joblib.load('heart_attack_model.pkl')  # Replace with your model file path

# st.sidebar.header('User Input Parameters')

# def user_input_features():
#     age = st.sidebar.slider('Age', 18, 100, 50)
#     heart_rate = st.sidebar.slider('Heart Rate', 40, 180, 70)
#     diabetes = st.sidebar.selectbox('Diabetes (0 for No, 1 for Yes)', [0, 1])
#     family_history = st.sidebar.selectbox('Family History (0 for No, 1 for Yes)', [0, 1])
#     smoking = st.sidebar.selectbox('Smoking (0 for No, 1 for Yes)', [0, 1])
#     obesity = st.sidebar.selectbox('Obesity (0 for No, 1 for Yes)', [0, 1])
#     alcohol_consumption = st.sidebar.selectbox('Alcohol Consumption (0 for No, 1 for Yes)', [0, 1])
#     exercise_hours_per_week = st.sidebar.slider('Exercise Hours Per Week', 0, 20, 3)
#     previous_heart_problems = st.sidebar.selectbox('Previous Heart Problems (0 for No, 1 for Yes)', [0, 1])
#     stress_level = st.sidebar.slider('Stress Level', 0, 10, 5)
#     sedentary_hours_per_day = st.sidebar.slider('Sedentary Hours Per Day', 0, 12, 5)
#     bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
#     triglycerides = st.sidebar.slider('Triglycerides', 50, 400, 150)
#     sleep_hours_per_day = st.sidebar.slider('Sleep Hours Per Day', 4, 12, 7)
#     systolic = st.sidebar.slider('Systolic BP', 90, 180, 120)
#     diastolic = st.sidebar.slider('Diastolic BP', 60, 120, 80)

#     # Checkbox inputs for categorical data
#     sex_female = st.sidebar.checkbox('Female')
#     sex_male = st.sidebar.checkbox('Male')
#     diet_average = st.sidebar.checkbox('Average Diet')
#     diet_healthy = st.sidebar.checkbox('Healthy Diet')
#     diet_unhealthy = st.sidebar.checkbox('Unhealthy Diet')
#     cholesterol_normal = st.sidebar.checkbox('Normal Cholesterol')
#     cholesterol_at_risk = st.sidebar.checkbox('Cholesterol At Risk')
#     cholesterol_high = st.sidebar.checkbox('High Cholesterol')
#     cholesterol_dangerous = st.sidebar.checkbox('Dangerous Cholesterol')

#     data = {
#         'age': Age,
#         'heart_rate': Heart Rate,
#         'diabetes': Diabetes,
#         'family_history': Family History,
#         'smoking': Smoking,
#         'obesity': Obesity,
#         'alcohol_consumption': Alcohol Consumption,
#         'exercise_hours_per_week': Exercise Hours Per Week,
#         'previous_heart_problems': Previous Heart Problems,
#         'stress_level': Stress Level,
#         'sedentary_hours_per_day': Sedentary Hours Per Day,
#         'bmi': BMI,
#         'triglycerides': Triglycerides,
#         'sleep_hours_per_day': Sleep Hours Per Day,
#         'systolic': Systolic,
#         'diastolic': Diastolic,
#         'sex_female': 1 if Sex_Female else 0,
#         'sex_male': 1 if Sex_Male else 0,
#         'diet_average': 1 if Diet_Average else 0,
#         'diet_healthy': 1 if Diet_Healthy else 0,
#         'diet_unhealthy': 1 if Diet_Unhealthy else 0,
#         'cholesterol_normal': 1 if Cholesterol_Normal else 0,
#         'cholesterol_at_risk': 1 if Cholesterol_At Risk else 0,
#         'cholesterol_high': 1 if Cholesterol_High else 0,
#         'cholesterol_dangerous': 1 if Cholesterol_Dangerous else 0
#     }

#     return pd.DataFrame(data, index=[0])

# df = user_input_features()

# st.subheader('User Input Parameters')
# st.write(df)

# # Make prediction
# prediction = GBC.predict(df)
# prediction_proba = GBC.predict_proba(df)

# st.subheader('Prediction')
# st.write('Heart Attack Risk:', 'High' if prediction[0] == 1 else 'Low')

# st.subheader('Prediction Probability')
# st.write(f"Probability of High Risk: {prediction_proba[0][1]:.2f}")
# st.write(f"Probability of Low Risk: {prediction_proba[0][0]:.2f}")



import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load your trained model (ensure the path is correct)
GBC = joblib.load('heart_attack_model.pkl')  # Replace with your model file path

st.sidebar.header('User Input Parameters')

def user_input_features():
    # Streamlit sidebar inputs
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

    # Ensure dictionary keys match exactly with dataset column names (with spaces included)
    data = {
        'Age': age,
        'Heart Rate': heart_rate,
        'Diabetes': diabetes,
        'Family History': family_history,
        'Smoking': smoking,
        'Obesity': obesity,
        'Alcohol Consumption': alcohol_consumption,
        'Exercise Hours Per Week': exercise_hours_per_week,
        'Previous Heart Problems': previous_heart_problems,
        'Stress Level': stress_level,
        'Sedentary Hours Per Day': sedentary_hours_per_day,
        'BMI': bmi,
        'Triglycerides': triglycerides,
        'Sleep Hours Per Day': sleep_hours_per_day,
        'Systolic BP': systolic,
        'Diastolic BP': diastolic,
        'Sex_Female': 1 if sex_female else 0,
        'Sex_Male': 1 if sex_male else 0,
        'Diet_Average': 1 if diet_average else 0,
        'Diet_Healthy': 1 if diet_healthy else 0,
        'Diet_Unhealthy': 1 if diet_unhealthy else 0,
        'Cholesterol_Normal': 1 if cholesterol_normal else 0,
        'Cholesterol_At Risk': 1 if cholesterol_at_risk else 0,
        'Cholesterol_High': 1 if cholesterol_high else 0,
        'Cholesterol_Dangerous': 1 if cholesterol_dangerous else 0
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

