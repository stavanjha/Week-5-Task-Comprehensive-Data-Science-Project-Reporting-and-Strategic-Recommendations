"""
Week 5 - Comprehensive Data Science Project
Breast Cancer Diagnostic Classification

Reproduces the core analysis used in the Week 5 report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

RANDOM_STATE = 42
FIG_DIR = "week5_figures"
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Load dataset
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
X = df.drop(columns=["target"])
y = df["target"]
target_names = list(data.target_names)

# 2. Explore dataset
print("Dataset shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Class counts:\n", y.value_counts())

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# 4. Scale data for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Define models
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        X_train_scaled, X_test_scaled
    ),
    "Decision Tree": (
        DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        X_train, X_test
    )
}

# 6. Train and evaluate
for name, (model, Xtr, Xte) in models.items():
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]

    print("\n===", name, "===")
    print("Accuracy :", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall   :", recall_score(y_test, pred))
    print("F1       :", f1_score(y_test, pred))
    print("ROC-AUC  :", roc_auc_score(y_test, proba))
    print(classification_report(y_test, pred, target_names=target_names))

    cv = cross_val_score(model, Xtr, y_train, cv=5, scoring="accuracy")
    print("5-fold CV:", cv.mean(), "+/-", cv.std())

# 7. Strategic comparison
print("\nRecommended benchmark: Logistic Regression")
print("Reason: strongest test metrics and fewer false negatives in this analysis.")

# 8. Optional extension
# Next steps: Random Forest, Gradient Boosting, PCA, threshold tuning,
# precision-recall analysis, external validation and calibration.
