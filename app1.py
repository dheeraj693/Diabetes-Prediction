import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to check if diabetic or not.")

# User input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
age = st.number_input("Age", min_value=0, max_value=120, step=1)


# User se input liya
#insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)

# Insulin_log calculate kiya
insulin_log = np.log(insulin + 1)   # +1 taaki zero case me error na ho

# Ab final features banaye
user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                       insulin, bmi, dpf, age, insulin_log]])

# Scale data
user_data_scaled = scaler.transform(user_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(user_data_scaled)
    if prediction[0] == 1:
        st.error("⚠️ The patient is likely Diabetic.")
    else:
        st.success("✅ The patient is NOT Diabetic.")
