"""
app.py

Deployment layer only. NO training happens here.

Loads:
    - model.joblib      (final trained Logistic Regression)
    - scaler.joblib      (StandardScaler fitted on training data)
    - encoders.joblib    (LabelEncoders for gender, smoking_history)
    - metadata.joblib    (feature order, final threshold, valid ranges)

Pipeline:
    User Input -> same LabelEncoders -> same StandardScaler -> same Model
                -> probability -> final threshold (0.3) -> Prediction
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="centered")

# ---------------------------------------------------------------
# Load saved artifacts (cached so they load once, not per click)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    encoders = joblib.load("encoders.joblib")
    metadata = joblib.load("metadata.joblib")
    return model, scaler, encoders, metadata

model, scaler, encoders, metadata = load_artifacts()

FEATURE_NAMES = metadata["feature_names"]
THRESHOLD = metadata["threshold"]
GENDER_CLASSES = metadata["gender_classes"]
SMOKING_CLASSES = metadata["smoking_classes"]
RANGES = metadata["feature_ranges"]

# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("Diabetes Prediction System")
st.caption(
    "This system uses a Machine Learning model (Logistic Regression) "
    "trained on patient health data to estimate the likelihood of diabetes."
)

st.divider()
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", GENDER_CLASSES)
    age = st.number_input(
        "Age", min_value=float(RANGES["age"][0]), max_value=float(RANGES["age"][1]),
        value=float(round((RANGES["age"][0] + RANGES["age"][1]) / 2)), step=1.0
    )
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

with col2:
    smoking_history = st.selectbox("Smoking History", SMOKING_CLASSES)
    bmi = st.number_input(
        "BMI", min_value=float(RANGES["bmi"][0]), max_value=float(RANGES["bmi"][1]),
        value=float(round((RANGES["bmi"][0] + RANGES["bmi"][1]) / 2, 2)), step=0.1
    )
    hba1c = st.number_input(
        "HbA1c Level", min_value=float(RANGES["HbA1c_level"][0]), max_value=float(RANGES["HbA1c_level"][1]),
        value=float(round((RANGES["HbA1c_level"][0] + RANGES["HbA1c_level"][1]) / 2, 1)), step=0.1
    )
    blood_glucose = st.number_input(
        "Blood Glucose Level", min_value=int(RANGES["blood_glucose_level"][0]),
        max_value=int(RANGES["blood_glucose_level"][1]),
        value=int((RANGES["blood_glucose_level"][0] + RANGES["blood_glucose_level"][1]) // 2), step=1
    )

st.divider()

if st.button("Predict", type="primary", use_container_width=True):

    # -----------------------------------------------------------
    # Build input row in EXACT same feature order as training
    # -----------------------------------------------------------
    input_dict = {
        "gender": encoders["gender"].transform([gender])[0],
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "smoking_history": encoders["smoking_history"].transform([smoking_history])[0],
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": blood_glucose,
    }

    input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]  # enforce training column order

    # Same scaler used during training — transform only, never fit
    input_scaled = scaler.transform(input_df)

    # Same final model — probability, then apply the final threshold
    probability = model.predict_proba(input_scaled)[0, 1]
    prediction = 1 if probability >= THRESHOLD else 0

    st.subheader("Result")

    r1, r2, r3 = st.columns(3)
    with r1:
        if prediction == 1:
            st.error("**Prediction**\n\nDiabetes")
        else:
            st.success("**Prediction**\n\nNo Diabetes")
    with r2:
        st.metric("Probability", f"{probability * 100:.1f}%")
    with r3:
        st.metric("Decision Threshold", f"{THRESHOLD * 100:.0f}%")

    st.caption("This prediction is for educational purposes and is not a medical diagnosis.")
