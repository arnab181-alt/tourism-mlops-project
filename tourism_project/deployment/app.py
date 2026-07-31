"""Streamlit frontend for the Wellness Tourism Package purchase predictor.

Streamlit Community Cloud runs this file directly from the repository. It loads
the trained model committed by the pipeline and returns a prediction for the
customer details entered in the form.
"""
import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Visit with Us - Wellness Tourism Package Predictor")
st.write(
    "Enter a customer's details to predict whether they are likely to purchase "
    "the newly introduced Wellness Tourism Package."
)

col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", 18, 100, 35)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", 0, 200, 15)
    Occupation = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
    NumberOfFollowups = st.number_input("Number of Followups", 0, 10, 3)
    MonthlyIncome = st.number_input("Monthly Income", 1000, 200000, 22000)
with col2:
    ProductPitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    NumberOfTrips = st.number_input("Number of Trips (per year)", 0, 50, 3)
    Passport = st.selectbox("Has Passport", [0, 1])
    PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    OwnCar = st.selectbox("Owns Car", [0, 1])
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", 0, 10, 0)
    Designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )

input_df = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.success(f"Likely to PURCHASE the package (probability {probability:.2%}).")
    else:
        st.info(f"Unlikely to purchase the package (probability {probability:.2%}).")
