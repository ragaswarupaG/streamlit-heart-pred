import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor

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

rf_hyp = RandomForestRegressor(max_depth = 15,random_state = 7, n_estimators=500, min_samples_leaf=2)
rf_hyp.fit(X_train, y_train)


st.write("""
# Heart Disease Prediction App
This app predicts the **presence of heart disease** based on user input parameters.
""")

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 50)
    heart_rate = st.sidebar.slider('Heart Rate', 60, 200, 80)
    diabetes = st.sidebar.selectbox('Diabetes', ['No', 'Yes'])
    family_history = st.sidebar.selectbox('Family History of Heart Disease', ['No', 'Yes'])
    smoking = st.sidebar.selectbox('Smoking', ['No', 'Yes'])
    obesity = st.sidebar.selectbox('Obesity', ['No', 'Yes'])
    alcohol_consumption = st.sidebar.selectbox('Alcohol Consumption', ['No', 'Yes'])
    exercise_hours_per_week = st.sidebar.slider('Exercise Hours Per Week', 0, 40, 10)
    previous_heart_problems = st.sidebar.selectbox('Previous Heart Problems', ['No', 'Yes'])
    stress_level = st.sidebar.slider('Stress Level (0-10)', 0, 10, 5)
    sedentary_hours_per_day = st.sidebar.slider('Sedentary Hours Per Day', 0, 24, 8)
    bmi = st.sidebar.slider('BMI', 10, 50, 25)
    triglycerides = st.sidebar.slider('Triglycerides', 50, 500, 150)
    sleep_hours_per_day = st.sidebar.slider('Sleep Hours Per Day', 0, 12, 7)
    systolic = st.sidebar.slider('Systolic Blood Pressure', 90, 200, 120)
    diastolic = st.sidebar.slider('Diastolic Blood Pressure', 60, 120, 80)

    sex = st.sidebar.radio('Sex', ['Male', 'Female'])
    diet = st.sidebar.selectbox('Diet', ['Average', 'Healthy', 'Unhealthy'])
    cholesterol = st.sidebar.selectbox('Cholesterol Level', ['Normal', 'At Risk', 'High', 'Dangerous'])


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

# Predict on user input
predicted_percentage = rf_hyp.predict(user_input)  # Predict the percentage risk
predicted_class = 1 if predicted_percentage >= 0.5 else 0  # Convert to binary classification

# Display results
st.subheader('Prediction')
st.write(f"Predicted Percentage Risk: {predicted_percentage[0]:.2f}")
st.write(f"Predicted Class: {predicted_class} (0 = No Heart Disease, 1 = Heart Disease)")



#     features = pd.DataFrame(data, index=[0])
#     return pd.DataFrame(data, index=[0])
#     return features



# # Get user input
# user_input = user_input_features()

# # Display user input
# st.subheader('User Input Parameters')
# st.write(user_input)

# RFR_predictions = (rf_hyp.predict(X_test) >= 0.5).astype(int)  


# # Display results
# st.subheader('Prediction')
# st.write('0 = No Heart Disease, 1 = Heart Disease')
# st.write(prediction)

