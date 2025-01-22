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

st.write("""
# Heart Disease Prediction App
This app predicts the **presence of heart disease** based on user input parameters.
""")

# Sidebar for user input
st.sidebar.header('User Input Parameters')

# def user_input_features():
#     age = st.sidebar.slider('Age', 18, 100, 50)
#     heart_rate = st.sidebar.slider('Heart Rate', 60, 200, 80)
#     diabetes = st.sidebar.selectbox('Diabetes', (0, 1))  # 0 = No, 1 = Yes
#     family_history = st.sidebar.selectbox('Family History', (0, 1))  # 0 = No, 1 = Yes
#     smoking = st.sidebar.selectbox('Smoking', (0, 1))  # 0 = No, 1 = Yes
#     obesity = st.sidebar.selectbox('Obesity', (0, 1))  # 0 = No, 1 = Yes
#     alcohol_consumption = st.sidebar.slider('Alcohol Consumption', 0, 30, 5)
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


def user_input_features():
    # Basic Information
    age = st.sidebar.slider('Age', 18, 100, 50)
    sex = st.sidebar.radio('Sex', ['Female', 'Male'])
    bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
    heart_rate = st.sidebar.slider('Heart Rate', 40, 180, 70)

    # Lifestyle Factors
    smoking = st.sidebar.radio('Smoking', ['No', 'Yes'])
    alcohol_consumption = st.sidebar.radio('Alcohol Consumption', ['No', 'Yes'])
    exercise_hours_per_week = st.sidebar.slider('Exercise Hours Per Week', 0, 20, 3)
    sedentary_hours_per_day = st.sidebar.slider('Sedentary Hours Per Day', 0, 12, 5)
    sleep_hours_per_day = st.sidebar.slider('Sleep Hours Per Day', 4, 12, 7)
    stress_level = st.sidebar.slider('Stress Level', 0, 10, 5)

    # Medical History
    diabetes = st.sidebar.radio('Diabetes', ['No', 'Yes'])
    family_history = st.sidebar.radio('Family History of Heart Disease', ['No', 'Yes'])
    previous_heart_problems = st.sidebar.radio('Previous Heart Problems', ['No', 'Yes'])
    cholesterol = st.sidebar.selectbox(
        'Cholesterol Level',
        ['Normal', 'At Risk', 'High', 'Dangerous']
    )
    diet = st.sidebar.selectbox(
        'Diet Type',
        ['Average', 'Healthy', 'Unhealthy']
    )

    # Convert categorical inputs to numerical values
    data = {
        'age': age,
        'sex_female': 1 if sex == 'Female' else 0,
        'sex_male': 1 if sex == 'Male' else 0,
        'bmi': bmi,
        'heart_rate': heart_rate,
        'smoking': 1 if smoking == 'Yes' else 0,
        'alcohol_consumption': 1 if alcohol_consumption == 'Yes' else 0,
        'exercise_hours_per_week': exercise_hours_per_week,
        'sedentary_hours_per_day': sedentary_hours_per_day,
        'sleep_hours_per_day': sleep_hours_per_day,
        'stress_level': stress_level,
        'diabetes': 1 if diabetes == 'Yes' else 0,
        'family_history': 1 if family_history == 'Yes' else 0,
        'previous_heart_problems': 1 if previous_heart_problems == 'Yes' else 0,
        'cholesterol_normal': 1 if cholesterol == 'Normal' else 0,
        'cholesterol_at_risk': 1 if cholesterol == 'At Risk' else 0,
        'cholesterol_high': 1 if cholesterol == 'High' else 0,
        'cholesterol_dangerous': 1 if cholesterol == 'Dangerous' else 0,
        'diet_average': 1 if diet == 'Average' else 0,
        'diet_healthy': 1 if diet == 'Healthy' else 0,
        'diet_unhealthy': 1 if diet == 'Unhealthy' else 0,
    }


    return pd.DataFrame(data, index=[0])
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

st.subheader('Prediction Probability')
st.write(prediction_proba)

