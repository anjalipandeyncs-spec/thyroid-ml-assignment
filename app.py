import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Thyroid Disease Prediction", layout="wide")

st.title("Thyroid Disease Prediction App")

# -------------------------------
# Load feature names
# -------------------------------
feature_path = "model/feature_names.pkl"

if not os.path.exists(feature_path):
    st.error("feature_names.pkl not found. Please run training again.")
    st.stop()

feature_names = joblib.load(feature_path)

# -------------------------------
# Model selection
# -------------------------------
model_files = {
    "Logistic Regression": "model/Logistic Regression.pkl",
    "Decision Tree": "model/Decision Tree.pkl",
    "KNN": "model/KNN.pkl",
    "Naive Bayes": "model/Naive Bayes.pkl",
    "Random Forest": "model/Random Forest.pkl",
    "XGBoost": "model/XGBoost.pkl"
}

model_choice = st.selectbox("Select Model", list(model_files.keys()))

model_path = model_files[model_choice]

if not os.path.exists(model_path):
    st.error(f"{model_choice} model not found.")
    st.stop()

model = joblib.load(model_path)

# -------------------------------
# File upload
# -------------------------------
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # -------------------------------
    # Handle target column
    # -------------------------------
    target_col = None
    possible_targets = ["target", "Class", "class"]

    for col in possible_targets:
        if col in data.columns:
            target_col = col
            break

    if target_col:
        y_true = data[target_col]
        X = data.drop(columns=[target_col])
    else:
        y_true = None
        X = data.copy()

    # -------------------------------
    # Align columns with training data
    # -------------------------------
    X = pd.get_dummies(X)

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names]

    # -------------------------------
    # Prediction
    # -------------------------------
    predictions = model.predict(X)

    data["Prediction"] = predictions

    st.subheader("Predictions")
    st.dataframe(data.head())

    # -------------------------------
    # Evaluation metrics
    # -------------------------------
    if y_true is not None:
        st.subheader("Evaluation Metrics")

        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, average='weighted')
        recall = recall_score(y_true, predictions, average='weighted')
        f1 = f1_score(y_true, predictions, average='weighted')

        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1-score: {f1:.4f}")

        # -------------------------------
        # Confusion matrix
        # -------------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, predictions)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # -------------------------------
        # Classification report
        # -------------------------------
        st.subheader("Classification Report")
        report = classification_report(y_true, predictions)
        st.text(report)

    else:
        st.info("No target column found. Only predictions displayed.")

