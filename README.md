# Thyroid Classification using Multiple ML Models

## Problem Statement
The goal of this project is to predict thyroid condition using patient medical features.  
Multiple machine learning classification models are implemented and compared.

---

## Dataset Description
- Dataset: Thyroid Dataset
- Instances: 9171
- Features: 22 medical features
- Target: Thyroid condition (class)
- Type: Classification problem

The dataset satisfies assignment constraints:
- Minimum instances required: 500
- Minimum features required: 12

---

## Models Used
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Naive Bayes
5. Random Forest
6. XGBoost

---

## Model Comparison Table
(Generated after running `train_and_save_models.py`)

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|----------|-----|-----------|--------|----|----|
| Logistic Regression | See model_results.csv |
| Decision Tree | See model_results.csv |
| KNN | See model_results.csv |
| Naive Bayes | See model_results.csv |
| Random Forest | See model_results.csv |
| XGBoost | See model_results.csv |

---

## Observations
- Logistic Regression provides a strong baseline.
- Decision Tree may slightly overfit.
- KNN is sensitive to feature scaling.
- Naive Bayes is fast but assumes feature independence.
- Random Forest provides stable and strong performance.
- XGBoost generally gives the best overall accuracy.

---

## Streamlit App Features
- CSV upload for test data
- Model selection dropdown
- Prediction output display
