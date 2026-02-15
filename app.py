import streamlit as st
import pandas as pd
import joblib

st.title("Thyroid Classification App")

uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    scaler = joblib.load("model/scaler.pkl")
    model = joblib.load(f"model/{model_name}.pkl")

    data_scaled = scaler.transform(data)
    predictions = model.predict(data_scaled)

    st.write("Predictions:")
    st.write(predictions)
