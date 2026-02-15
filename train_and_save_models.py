import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

# -----------------------
# 1. Load Dataset
# -----------------------
df = pd.read_csv("Thyroid-Dataset.csv")

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert numeric columns
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Fill missing values
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Target handling
if 'class_negative' in df.columns:
    X = df.drop('class_negative', axis=1)
    y = df['class_negative']
else:
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

# -----------------------
# 2. Train-test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# -----------------------
# 3. Models
# -----------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss', n_estimators=50, max_depth=4)
}

results = []

# -----------------------
# 4. Train + Evaluate
# -----------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = roc_auc_score(y_test, y_pred)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1": f1_score(y_test, y_pred, average='weighted'),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    results.append(metrics)

    # Save model
    joblib.dump(model, f"model/{name}.pkl")

# -----------------------
# 5. Save results
# -----------------------
results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)

print("\nModel Comparison Results:\n")
print(results_df.round(4))
