import streamlit as st
import joblib
import pandas as pd

from features import create_features  # your big feature file

# =========================
# LOAD MODEL (PIPELINE)
# =========================
model = joblib.load("Model/quora_final_pipeline.pkl")

# =========================
# UI
# =========================
st.set_page_config(page_title="Duplicate Question Detector")

st.title("🔍 Duplicate Question Detector")
st.caption("Check if two questions are semantically similar")

# =========================
# INPUT
# =========================
q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

# =========================
# PREDICTION
# =========================
if st.button("Check Similarity"):

    if q1.strip() == "" or q2.strip() == "":
        st.warning("Please enter both questions")
    else:
        # Create features
        df = create_features(q1, q2)
        # Predict
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        # Output
        if pred == 1:
            st.success(f"✅ Duplicate Questions (Confidence: {round(prob, 2)})")
        else:
            st.error(f"❌ Not Duplicate (Confidence: {round(1 - prob, 2)})")