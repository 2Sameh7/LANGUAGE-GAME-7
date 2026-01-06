# -*- coding: utf-8 -*-
# Loan Dataset Task 4 - Template
# عدّل القيم (TODO) حسب رقمك الجامعي والهايبر بارامترز المرسلة لك

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay

# -----------------------------
# 1) قراءة الداتاست
# -----------------------------
df = pd.read_csv("loan_dataset.csv")  # ضع مسار داتاستك الصحيحة

print("Shape:", df.shape)
print(df.head())

# -----------------------------
# 2) تحديد الأعمدة
# -----------------------------
numeric_cols = [
    'person_age','person_income','person_emp_exp','loan_amnt',
    'loan_int_rate','loan_percent_income','cb_person_cred_hist_length',
    'credit_score'
]
categorical_cols = [
    'person_gender','person_education','person_home_ownership',
    'loan_intent','previous_loan_defaults_on_file'
]
target = 'loan_status'

# -----------------------------
# 3) معالجة القيم المفقودة
# -----------------------------
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -----------------------------
# 4) تقسيم البيانات
# -----------------------------
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# 5) تجهيز الترميز والمعايير
# -----------------------------
preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# -----------------------------
# 6) نموذج Logistic Regression
# -----------------------------
log_reg = Pipeline(steps=[
    ('prep', preprocess),
    ('clf', LogisticRegression(
        C=1.0,            # TODO: ضع القيمة المرتبطة برقمك الجامعي
        penalty='l2',     # TODO: برر اختيارك
        solver='lbfgs',
        max_iter=2000,
        class_weight=None # أو 'balanced' إذا في عدم توازن
    ))
])

log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
y_proba_lr = log_reg.predict_proba(X_test)[:,1]

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1:", f1_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_lr))

# -----------------------------
# 7) نموذج RandomForest
# -----------------------------
rf = Pipeline(steps=[
    ('prep', preprocess),
    ('clf', RandomForestClassifier(
        n_estimators=150,     # TODO: عدّل حسب رقمك الجامعي
        max_depth=10,         # TODO
        min_samples_split=4,  # TODO
        min_samples_leaf=2,   # TODO
        max_features='sqrt',
        random_state=42,
        class_weight=None
    ))
])

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:,1]

print("\nRandomForest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1:", f1_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))

# -----------------------------
# 8) مصفوفة الالتباس + ROC
# -----------------------------
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - RandomForest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

RocCurveDisplay.from_predictions(y_test, y_proba_rf)
plt.title("ROC Curve - RandomForest")
plt.show()