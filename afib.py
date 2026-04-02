import streamlit as st
import numpy as np
import pandas as pd
import pickle

def load_model():
    with open("wv_abcb.pkl", "rb") as f:
        return pickle.load(f)

def user_input():
    with st.form("afib_form"):
        st.markdown("### Basic Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            SEX = st.radio("Gender", ["Male", "Female"])
        with col2:
            INCOME = st.selectbox("Income Level", ["Low", "Middle", "High"])
        with col3:
            REGION = st.radio("Region", ["Urban", "Rural"])

        st.markdown("### Lifestyle & History")
        col4, col5 = st.columns(2)
        with col4:
            Age_group = st.selectbox("Age Group", [
                "20–24", "25–29", "30–34", "35–39", "40–44",
                "45–49", "50–54", "55–59", "60–64", "65–69",
                "70–74", "75–79", "80–84", "85+"
            ])
            EXERCISE = st.radio(
                "Physical activity sessions per week",
                ["0-2 times", "3-4 times", "5-6 times", "Everyday"]
            )
        with col5:
            SMK = st.radio("Smoking Status", ["Never smoked", "Current smoker"])
            DRNK = st.radio("Alcoholic drinks per week", ["Rarely", "Sometimes", "Everyday"])
            M_HRT = st.radio("Heart Disease Diagnosis", ["No", "Yes"])

        st.markdown("### Clinical Measurements")
        col6, col7 = st.columns(2)
        with col6:
            BP_HIGH = st.slider("Systolic Blood Pressure (mmHg)", 70, 250, 120)
            BP_LWST = st.slider("Diastolic Blood Pressure (mmHg)", 40, 140, 77)
            BLDS = st.slider("Fasting Blood Glucose (mg/dL)", 50, 400, 91)
            TOT_CHOLE = st.slider("Total Cholesterol (mg/dL)", 80, 400, 189)
            HMG = st.slider("Hemoglobin (g/dL)", 7, 20, 14)
        with col7:
            SGOT_AST = st.slider("SGOT (AST) (IU/L)", 5, 400, 23)
            SGPT_ALT = st.slider("SGPT (ALT) (IU/L)", 5, 400, 22)
            GAMMA_GTP = st.slider("Gamma-GTP (IU/L)", 10, 700, 28)
            BMI = st.slider("Body Mass Index (BMI)", 12, 50, 23)

        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    age_mapping = {
        "20–24": 5, "25–29": 6, "30–34": 7, "35–39": 8,
        "40–44": 9, "45–49": 10, "50–54": 11, "55–59": 12,
        "60–64": 13, "65–69": 14, "70–74": 15, "75–79": 16,
        "80–84": 17, "85+": 18
    }

    AGE = age_mapping[Age_group]
    INCOME = {"Low": 1, "Middle": 2, "High": 3}[INCOME]
    EXERCISE = {"0-2 times": 0, "3-4 times": 1, "5-6 times": 2, "Everyday": 3}[EXERCISE]
    SMK = {"Never smoked": 0, "Current smoker": 1}[SMK]
    DRNK = {"Rarely": 0, "Sometimes": 1, "Everyday": 2}[DRNK]
    M_HRT = {"No": 0, "Yes": 1}[M_HRT]
    SEX_2_0 = 1 if SEX == "Female" else 0
    REGION_1_0 = 1 if REGION == "Urban" else 0

    data = {
        "AGE": AGE,
        "INCOME": INCOME,
        "BP_HIGH": BP_HIGH,
        "BP_LWST": BP_LWST,
        "BLDS": BLDS,
        "TOT_CHOLE": TOT_CHOLE,
        "HMG": HMG,
        "SGOT_AST": SGOT_AST,
        "SGPT_ALT": SGPT_ALT,
        "GAMMA_GTP": GAMMA_GTP,
        "SMK": SMK,
        "DRNK": DRNK,
        "EXERCISE": EXERCISE,
        "M_HRT": M_HRT,
        "BMI": BMI,
        "REGION_1.0": REGION_1_0,
        "SEX_2.0": SEX_2_0
    }

    return submitted, pd.DataFrame([data])

def predict(input_df):
    artifact = load_model()
    wv = artifact["WeightedVoting"]

    ab_model = wv["ab_model"]
    cb_model = wv["cb_model"]
    ab_weight = wv["ab_weight"]
    cb_weight = wv["cb_weight"]
    threshold = wv["threshold"]

    ab_proba = ab_model.predict_proba(input_df)[:, 1]
    cb_proba = cb_model.predict_proba(input_df)[:, 1]

    wv_proba = cb_weight * cb_proba + ab_weight * ab_proba

    return wv_proba[0], threshold

def main():
    st.set_page_config(page_title="A-fib Risk Predictor", page_icon="🫀", layout="centered")

    st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .hero-box {
            padding: 1.2rem 1.4rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
            border: 1px solid #dbe7ff;
            margin-bottom: 1rem;
        }
        .result-box {
            padding: 1.2rem;
            border-radius: 16px;
            border: 1px solid #e6e6e6;
            background-color: #fafafa;
            margin-top: 1rem;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #ff4b4b, #ff6b6b) !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 12px !important;
            height: 3em !important;
            font-size: 16px !important;
            border: none !important;
            transition: 0.2s !important;
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #ff3b3b, #ff5b5b) !important;
            color: white !important;
            transform: scale(1.02);
        }
        div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(90deg, #ff4b4b, #ff6b6b) !important;
            color: white !important;
            font-weight: 700 !important;
            border-radius: 12px !important;
            height: 3.2em !important;
            font-size: 16px !important;
            border: none !important;
            width: 100% !important;
            transition: 0.2s !important;
        }
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: linear-gradient(90deg, #ff3b3b, #ff5b5b) !important;
            color: white !important;
            transform: scale(1.02);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="hero-box">
            <h1 style="margin-bottom:0.3rem;">🫀 A-fib Risk Prediction Web</h1>
            <p style="margin-bottom:0;">
                Enter health check-up information to estimate the future risk of atrial fibrillation.
            </p>
        </div>
    """, unsafe_allow_html=True)

    submitted, input_df = user_input()

    if submitted:
        prob, threshold = predict(input_df)

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Risk", f"{prob:.2%}")
        with col2:
            st.metric("Decision Threshold", f"{threshold:.2%}")

        progress_value = float(min(max(prob, 0.0), 1.0))
        st.progress(progress_value)

        if prob >= threshold:
            st.error("High risk detected. Medical consultation is recommended.")
        else:
            st.success("Risk appears relatively low. Continue healthy lifestyle management.")

if __name__ == "__main__":
    main()