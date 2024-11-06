import streamlit as st
import joblib

# Load the model
model = joblib.load('model.joblib')

# Mean & std 
mean = {'Gender': 0.49, 
        'Age': 37.655, 
        'EstimatedSalary': 69742.5
}

std = {'Gender': 0.5005260390723666, 
       'Age': 10.48287659730792, 
       'EstimatedSalary': 34096.960282424785
}

# Project description on web interface
st.title("Predicting Purchase Behavior on Social Network Ads Using Machine Learning")
st.header("Leveraging Machine Learning for Classifying User Response to Social Network Ads")
st.markdown("""
    This application uses advanced machine learning algorithms to predict User Response (Will_Purchase - Yes/ No)
    based on Gender, Age, & EstimatedSlary. After testing multiple models, we have selected the most reliable
    model to ensure accurate predictions with hyperparameter tuning and cross-validation for improved reliability.
    This will help to in target marketing as the company can reach out to potential leads for making customers. 
""")

# Metrics 
st.metric(label="K - Nearest Neighbour accuracy", value="95%", delta="Up 4.90001% from previous model")

# User Input 
gender = st.selectbox("Gender", ["Male", "Female"]) # Gender
gender_encoded = 1 if gender == "Male" else 0 # Convert the Gender to numerical encoding
age = st.number_input("Age", min_value=18, max_value=60, value=30) # Age 
salary = st.number_input("EstimatedSalary", min_value=15000, max_value=150000, value=50000) # Salary

# Standard Scaling function
def scale_input(input_data):
    # Using zip to directly pair each feature value with its mean and std_dev
    return [(value - mean[key]) / std[key] for value, key in zip(input_data, mean.keys())]

# Prediction
scaled_data = scale_input([gender_encoded, age, salary]) # Scaling the data
pred = model.predict([scaled_data]) 
result = "probably not purchase" if pred[0] == 0 else "probably purchase"

# Output 
if st.button("Predict"):
    st.success(f"The user with age {age}, gender {gender}, & salary {salary} will {result}, If your advertisement is displayed.")
