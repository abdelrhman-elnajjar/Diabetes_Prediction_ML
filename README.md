# Diabetes Prediction System
 
## Overview
 
This project uses Machine Learning to predict whether a patient is likely to have diabetes, based on health and demographic features such as age, BMI, blood glucose level, and HbA1c level.
 
## Problem
 
Diabetes is often diagnosed late, after complications have already started. Early identification of at-risk patients allows for timely intervention and lifestyle changes. This project explores whether routine clinical measurements can be used to build a predictive model that flags patients who may need further screening.
 
## Dataset
 
- Original records: 100,000
- Records after removing duplicates: 96,146
- 8 input features
- 1 target column: `diabetes`
**Features:**
 
- gender
- age
- hypertension
- heart_disease
- smoking_history
- bmi
- HbA1c_level
- blood_glucose_level
**Target:**
 
`diabetes`
 
- 0 = No Diabetes
- 1 = Diabetes
## Data Preprocessing
 
- Duplicate removal
- Missing-value checking
- Categorical encoding (`gender`, `smoking_history`) using LabelEncoder
- Stratified train/test split (70% / 30%)
- Feature scaling using StandardScaler
## Models
 
Two classical ML models were trained and compared using an identical preprocessing pipeline:
 
1. Logistic Regression
2. Linear SVM
## Threshold Tuning
 
The dataset is imbalanced (only ~8.5% of patients have diabetes), so relying on the default threshold of 0.5 caused the model to miss a significant number of real diabetes cases. Since Recall (catching real cases) matters more than Precision in a medical screening context, the decision threshold was tuned instead of using the default.
 
**Final threshold: 0.3**
 
## Results
 
Both models were evaluated at threshold = 0.3:
 
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 94.95% | 71.35% | 71.43% | 71.39% | 95.94% |
| Linear SVM | 94.93% | 71.65% | 70.49% | 71.06% | 95.76% |
 
**Logistic Regression** was selected as the final model. It achieved slightly higher Recall and ROC-AUC than Linear SVM, while training much faster — making it the more practical choice for this project.
 
## Deployment
 
The final model was deployed using **Streamlit**.
 
**Live Demo:** [https://diabetes-ml-project-2026.streamlit.app/](https://diabetes-ml-project-2026.streamlit.app/)
 
The application does **not** retrain the model every time it runs. The trained model and all preprocessing components are saved with `joblib` during a one-time training step and are simply loaded by the app on startup.
 
**Workflow:**
 
```
User Input
   ↓
Preprocessing (Label Encoding + StandardScaler)
   ↓
Saved Logistic Regression Model
   ↓
Probability
   ↓
Threshold = 0.3
   ↓
Prediction
```
 
The app has two pages:
 
- **Diabetes Prediction System** — the main prediction interface
- **Team** — project team and supervisors
## Project Structure
 
```
Diabetes_Prediction_ML/
├── Diabetes_Prediction_ML.ipynb   # Training, experiments, and evaluation
├── app.py                         # Streamlit app — main prediction page
├── pages/
│   └── 1_Team.py                  # Streamlit app — Team page
├── train_and_save.py              # One-time script: trains and saves model/scaler/encoders
├── model.joblib                   # Saved final Logistic Regression model
├── scaler.joblib                  # Saved StandardScaler
├── encoders.joblib                # Saved LabelEncoders (gender, smoking_history)
├── metadata.joblib                # Feature order, final threshold, valid input ranges
├── requirements.txt
├── README.md
└── diabetes_prediction_dataset.csv
```
 
## How to Run
 
Install the dependencies:
 
```
pip install -r requirements.txt
```
 
Run the Streamlit application:
 
```
streamlit run app.py
```
 
The saved model files are already included, so the app loads immediately without retraining. If you want to regenerate them from scratch, run `python train_and_save.py` first.
 
## Team
 
Abdelrhman Ehab Soliman and 
Mohamed Elmogy
 
**Supervised by:**
Dr. Kholoud Amer
Dr. Yasmin Mahmoud
 
## Disclaimer
 
This project is for educational purposes only and is not intended to provide medical diagnosis or replace professional medical advice.
 
## GitHub
 
## GitHub

GitHub Repository: https://github.com/abdelrhman-elnajjar/Diabetes_Prediction_ML
