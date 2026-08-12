"""
train_and_save.py

يشغّل هذا السكريبت *نفس* خطوات التدريب والـpreprocessing الموجودة بالضبط
في Diabetes_Prediction_ML.ipynb (نفس الترتيب، نفس البارامترات،
نفس random_state=42، نفس test_size=0.3 مع stratify=y، نفس StandardScaler،
نفس LabelEncoder لكل من gender و smoking_history).

الإضافة الوحيدة هنا هي حفظ الـfinal model (Logistic Regression) والـScaler
والـLabel Encoders باستخدام joblib، حتى يستخدمها app.py دون إعادة تدريب.

لا يتم تعديل الـNotebook الأصلي إطلاقًا — هذا السكريبت منفصل تمامًا،
يُشغَّل مرة واحدة فقط أثناء تجهيز الـDeployment.

الـfinal model المختار: Logistic Regression (بناءً على نتائج الـNotebook:
Recall = 71.43%, ROC-AUC = 95.94%, أسرع في التدريب من SVM).
الـfinal threshold: 0.3
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ---------------------------------------------------------------
# 1. Load the dataset (same as notebook Cell 3)
# ---------------------------------------------------------------
df = pd.read_csv('diabetes_prediction_dataset.csv')

# ---------------------------------------------------------------
# 2. Remove duplicate rows (same as notebook Cell 10)
# ---------------------------------------------------------------
df = df.drop_duplicates()

# ---------------------------------------------------------------
# 3. Preprocessing: Label Encoding (same as notebook Cell 16)
# ---------------------------------------------------------------
data = df.copy()

gender_encoder = LabelEncoder()
data['gender'] = gender_encoder.fit_transform(data['gender'])

smoking_encoder = LabelEncoder()
data['smoking_history'] = smoking_encoder.fit_transform(data['smoking_history'])

# ---------------------------------------------------------------
# 4. Features / target (same as notebook Cell 17)
# ---------------------------------------------------------------
X = data.drop('diabetes', axis=1)
y = data['diabetes']

feature_names = X.columns.tolist()

# ---------------------------------------------------------------
# 5. Train/test split (same as notebook Cell 19)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------------
# 6. Scaling (same as notebook Cell 20)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 7. Train Logistic Regression -> FINAL MODEL (same as notebook Cell 22)
# ---------------------------------------------------------------
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------
# 8. Sanity check: reproduce the notebook's reported metrics at t=0.3
#    (same as notebook Cells 23, 29) to confirm this is the same model.
# ---------------------------------------------------------------
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

FINAL_THRESHOLD = 0.3
y_pred_final = (y_pred_proba >= FINAL_THRESHOLD).astype(int)

print("=== Reproduced metrics (must match the notebook) ===")
print("Accuracy :", accuracy_score(y_test, y_pred_final))
print("Precision:", precision_score(y_test, y_pred_final))
print("Recall   :", recall_score(y_test, y_pred_final))
print("F1 Score :", f1_score(y_test, y_pred_final))
print("ROC-AUC  :", roc_auc_score(y_test, y_pred_proba))
print("======================================================")
print("Expected from notebook -> Accuracy: 0.9495, Precision: 0.7135, "
      "Recall: 0.7143, F1: 0.7139, ROC-AUC: 0.9594")

# ---------------------------------------------------------------
# 9. Save the final model, scaler, encoders, and metadata
# ---------------------------------------------------------------
joblib.dump(log_reg, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(
    {'gender': gender_encoder, 'smoking_history': smoking_encoder},
    'encoders.joblib'
)
joblib.dump(
    {
        'feature_names': feature_names,
        'threshold': FINAL_THRESHOLD,
        'gender_classes': gender_encoder.classes_.tolist(),
        'smoking_classes': smoking_encoder.classes_.tolist(),
        'feature_ranges': {
            'age': (float(df['age'].min()), float(df['age'].max())),
            'bmi': (float(df['bmi'].min()), float(df['bmi'].max())),
            'HbA1c_level': (float(df['HbA1c_level'].min()), float(df['HbA1c_level'].max())),
            'blood_glucose_level': (int(df['blood_glucose_level'].min()), int(df['blood_glucose_level'].max())),
        },
    },
    'metadata.joblib'
)

print("\nSaved: model.joblib, scaler.joblib, encoders.joblib, metadata.joblib")
