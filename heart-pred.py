import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection

# Load the preprocessed dataset
def load_data():
    df = pd.read_csv('preprocessed_heart_data.csv') 
    return df

df = load_data()

# Split the dataset into features (X) and target (y)
X = df.drop('Heart Attack Risk', axis=1) 
y = df['Heart Attack Risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30, random_state=7)

# Train a Gradient Boosting Classifier
GBC = GradientBoostingClassifier(
    n_estimators=1000,         
    learning_rate=0.05,       
    max_depth=3,             
    min_samples_split=15,     
    min_samples_leaf=5,     
    random_state=7
)
GBC.fit(X_train, y_train)

# st.write("""
# # Heart Disease Prediction App
# This app predicts the **presence of heart disease** based on user input parameters.
# """)

# # Sidebar for user input
# st.sidebar.header('User Input Parameters')

# def user_input_features():
#     age = st.sidebar.slider('Age', 18, 100, 50)
#     heart_rate = st.sidebar.slider('Heart Rate', 60, 200, 80)
#     diabetes = st.sidebar.selectbox('Diabetes', (0, 1))  # 0 = No, 1 = Yes
#     family_history = st.sidebar.selectbox('Family History', (0, 1))  # 0 = No, 1 = Yes
#     smoking = st.sidebar.selectbox('Smoking', (0, 1))  # 0 = No, 1 = Yes
#     obesity = st.sidebar.selectbox('Obesity', (0, 1))  # 0 = No, 1 = Yes
#     alcohol_consumption = st.sidebar.selectbox('Alcohol Consumption', (0, 1))  # 0 = No, 1 = Yes

#     exercise_hours_per_week = st.sidebar.slider('Exercise Hours Per Week', 0, 40, 10)
#     previous_heart_problems = st.sidebar.selectbox('Previous Heart Problems', (0, 1))  # 0 = No, 1 = Yes
#     stress_level = st.sidebar.slider('Stress Level', 0, 10, 5)
#     sedentary_hours_per_day = st.sidebar.slider('Sedentary Hours Per Day', 0, 24, 8)
#     bmi = st.sidebar.slider('BMI', 10, 50, 25)
#     triglycerides = st.sidebar.slider('Triglycerides', 50, 500, 150)
#     sleep_hours_per_day = st.sidebar.slider('Sleep Hours Per Day', 0, 12, 7)
#     systolic = st.sidebar.slider('Systolic', 90, 200, 120)
#     diastolic = st.sidebar.slider('Diastolic', 60, 120, 80)
#     sex_female = st.sidebar.selectbox('Sex (Female)', (0, 1))  # 0 = No, 1 = Yes
#     sex_male = st.sidebar.selectbox('Sex (Male)', (0, 1))  # 0 = No, 1 = Yes
#     diet_average = st.sidebar.selectbox('Diet (Average)', (0, 1))  # 0 = No, 1 = Yes
#     diet_healthy = st.sidebar.selectbox('Diet (Healthy)', (0, 1))  # 0 = No, 1 = Yes
#     diet_unhealthy = st.sidebar.selectbox('Diet (Unhealthy)', (0, 1))  # 0 = No, 1 = Yes
#     cholesterol_normal = st.sidebar.selectbox('Cholesterol (Normal)', (0, 1))  # 0 = No, 1 = Yes
#     cholesterol_at_risk = st.sidebar.selectbox('Cholesterol (At Risk)', (0, 1))  # 0 = No, 1 = Yes
#     cholesterol_high = st.sidebar.selectbox('Cholesterol (High)', (0, 1))  # 0 = No, 1 = Yes
#     cholesterol_dangerous = st.sidebar.selectbox('Cholesterol (Dangerous)', (0, 1))  # 0 = No, 1 = Yes

#     data = {
#         'Age': age,
#         'Heart Rate': heart_rate,
#         'Diabetes': diabetes,
#         'Family History': family_history,
#         'Smoking': smoking,
#         'Obesity': obesity,
#         'Alcohol Consumption': alcohol_consumption,
#         'Exercise Hours Per Week': exercise_hours_per_week,
#         'Previous Heart Problems': previous_heart_problems,
#         'Stress Level': stress_level,
#         'Sedentary Hours Per Day': sedentary_hours_per_day,
#         'BMI': bmi,
#         'Triglycerides': triglycerides,
#         'Sleep Hours Per Day': sleep_hours_per_day,
#         'Systolic': systolic,
#         'Diastolic': diastolic,
#         'Sex_Female': sex_female,
#         'Sex_Male': sex_male,
#         'Diet_Average': diet_average,
#         'Diet_Healthy': diet_healthy,
#         'Diet_Unhealthy': diet_unhealthy,
#         'Cholesterol_Normal': cholesterol_normal,
#         'Cholesterol_At Risk': cholesterol_at_risk,
#         'Cholesterol_High': cholesterol_high,
#         'Cholesterol_Dangerous': cholesterol_dangerous
#     }
#     features = pd.DataFrame(data, index=[0])





#     return pd.DataFrame(data, index=[0])
#     return features

# # Get user input
# user_input = user_input_features()

# # Display user input
# st.subheader('User Input Parameters')
# st.write(user_input)

# # Make predictions
# prediction = GBC.predict(user_input)
# prediction_proba = GBC.predict_proba(user_input)

# # Display results
# st.subheader('Prediction')
# st.write('0 = No Heart Disease, 1 = Heart Disease')
# st.write(prediction)







st.write("""
# Heart Disease Prediction App
This app predicts the **presence of heart disease** based on user input parameters.
""")

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.slider('Age', 18, 100, 50)
    heart_rate = st.slider('Heart Rate', 60, 200, 80)
    diabetes = st.selectbox('Diabetes', ['No', 'Yes'])
    family_history = st.selectbox('Family History of Heart Disease', ['No', 'Yes'])
    smoking = st.selectbox('Smoking', ['No', 'Yes'])
    obesity = st.selectbox('Obesity', ['No', 'Yes'])
    alcohol_consumption = st.selectbox('Alcohol Consumption', ['No', 'Yes'])
    exercise_hours_per_week = st.slider('Exercise Hours Per Week', 0, 40, 10)
    previous_heart_problems = st.selectbox('Previous Heart Problems', ['No', 'Yes'])
    stress_level = st.slider('Stress Level (0-10)', 0, 10, 5)
    sedentary_hours_per_day = st.slider('Sedentary Hours Per Day', 0, 24, 8)
    bmi = st.slider('BMI', 10, 50, 25)
    triglycerides = st.slider('Triglycerides', 50, 500, 150)
    sleep_hours_per_day = st.slider('Sleep Hours Per Day', 0, 12, 7)
    systolic = st.slider('Systolic Blood Pressure', 90, 200, 120)
    diastolic = st.slider('Diastolic Blood Pressure', 60, 120, 80)

    sex = st.radio('Sex', ['Male', 'Female'])
    diet = st.selectbox('Diet', ['Average', 'Healthy', 'Unhealthy'])
    cholesterol = st.selectbox('Cholesterol Level', ['Normal', 'At Risk', 'High', 'Dangerous'])


        
 
        
        
        
     
      

       
        

    data = {
        'Age': age,
        'Heart Rate': heart_rate,
        'Diabetes': 1 if diabetes == 'Yes' else 0,
        'Family History': 1 if family_history == 'Yes' else 0,
        'Smoking': 1 if smoking == 'Yes' else 0,
        'Obesity': 1 if obesity == 'Yes' else 0,
        'Alcohol Consumption': 1 if alcohol_consumption == 'Yes' else 0,
        'Exercise Hours Per Week': exercise_hours_per_week,
        'Previous Heart Problems': 1 if previous_heart_problems == 'Yes' else 0,
        'Stress Level': stress_level,
        'Sedentary Hours Per Day': sedentary_hours_per_day,
        'BMI': bmi,
        'Triglycerides': triglycerides,
        'Sleep Hours Per Day': sleep_hours_per_day,
        'Systolic': systolic,
        'Diastolic': diastolic,
        'Sex_Female': 1 if sex == 'Female' else 0,
        'Sex_Male': 1 if sex == 'Male' else 0,
        'Diet_Average': 1 if diet == 'Average' else 0,
        'Diet_Healthy': 1 if diet == 'Healthy' else 0,
        'Diet_Unhealthy': 1 if diet == 'Unhealthy' else 0,
        'Cholesterol_Normal': 1 if cholesterol == 'Normal' else 0,
        'Cholesterol_At Risk': 1 if cholesterol == 'At Risk' else 0,
        'Cholesterol_High': 1 if cholesterol == 'High' else 0,
        'Cholesterol_Dangerous': 1 if cholesterol == 'Dangerous' else 0
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
user_input = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(user_input)

# Make predictions
prediction = GBC.predict(user_input)
prediction_proba = GBC.predict_proba(user_input)

# Display results
st.subheader('Prediction')
st.write('0 = No Heart Disease, 1 = Heart Disease')
st.write(prediction)

# Display prediction probability
st.subheader('Prediction Probability')
st.write(prediction_proba)

