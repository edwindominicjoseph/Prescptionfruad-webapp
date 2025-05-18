import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import requests
from io import StringIO
import os

# --- Load Dataset ---

file_id = "1Il7nb24z1hktzJb6oX-Bj1U0-ikdJmT-"  # Replace with your actual file ID
url = f"https://drive.google.com/uc?export=download&id={file_id}"

response = requests.get(url)
df3 = pd.read_csv(StringIO(response.text))


df3 = df3.drop(columns=['PATIENT_enc', 'Id_patient', 'Id', 'HEALTHCARE_COVERAGE', 'TOTAL_CLAIM_COST'])
df3 = df3.rename(columns={'PATIENT_med': 'PATIENT_ID'})

# --- Medication Risk Categorization ---
high_risk_keywords = ["oxycodone", "hydrocodone", "fentanyl", "morphine", "codeine",
                      "tramadol", "methadone", "buprenorphine", "alprazolam", "diazepam",
                      "clonazepam", "midazolam", "sufentanil", "remifentanil",
                      "paclitaxel", "cisplatin", "doxorubicin", "methotrexate"]

moderate_risk_keywords = ["prednisone", "steroid", "tamoxifen", "warfarin", "lithium",
                          "immunosuppressant", "hormone", "propofol", "baricitinib", "insulin",
                          "cyclosporine", "glucocorticoid", "barbiturate"]

def categorize_risk(med_name):
    name = med_name.lower()
    if any(keyword in name for keyword in high_risk_keywords):
        return "High Risk"
    elif any(keyword in name for keyword in moderate_risk_keywords):
        return "Moderate Risk"
    else:
        return "Low Risk"

df3["MEDICATION_RISK"] = df3["DESCRIPTION_med"].apply(categorize_risk)
risk_mapping = {"Low Risk": 0, "Moderate Risk": 1, "High Risk": 2}
df3["MEDICATION_RISK_CODE"] = df3["MEDICATION_RISK"].map(risk_mapping)

# --- Train Medication Risk Classifier ---
X_risk = df3["DESCRIPTION_med"]
y_risk = df3["MEDICATION_RISK"]
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_risk)
risk_classifier = LogisticRegression(max_iter=1000)
risk_classifier.fit(X_vec, y_risk)

# --- Encode Categorical Features ---
encoded_df = df3.copy()
label_encoders = {}
categorical_cols = ['DESCRIPTION_med', 'ENCOUNTERCLASS', 'PROVIDER', 'ORGANIZATION', 'GENDER', 'ETHNICITY', 'MARITAL', 'STATE']
for col in categorical_cols:
    le = LabelEncoder()
    encoded_df[col + "_encoded"] = le.fit_transform(encoded_df[col])
    label_encoders[col] = le

# --- General Model ---
feature_cols = ['DESCRIPTION_med_encoded', 'ENCOUNTERCLASS_encoded', 'DISPENSES', 'TOTALCOST', 'AGE', 'MEDICATION_RISK_CODE']
general_model = IsolationForest(contamination=0.05, random_state=42)
general_model.fit(encoded_df[feature_cols])
general_scores = general_model.decision_function(encoded_df[feature_cols])
general_min_score, general_max_score = general_scores.min(), general_scores.max()

# --- Patient History (Delta Model) ---
patient_history = df3.groupby("PATIENT_ID").agg({
    "BASE_COST": "mean",
    "TOTALCOST": "mean",
    "DISPENSES": "mean",
    "AGE": "first"
}).reset_index()

delta_data = df3[["PATIENT_ID", "BASE_COST", "TOTALCOST", "DISPENSES"]].merge(
    patient_history, on="PATIENT_ID", suffixes=("", "_avg"))

delta_data["delta_base"] = delta_data["BASE_COST"] - delta_data["BASE_COST_avg"]
delta_data["delta_cost"] = delta_data["TOTALCOST"] - delta_data["TOTALCOST_avg"]
delta_data["delta_disp"] = delta_data["DISPENSES"] - delta_data["DISPENSES_avg"]
delta_features = delta_data[["delta_base", "delta_cost", "delta_disp"]]

delta_model = IsolationForest(contamination=0.05, random_state=42)
delta_model.fit(delta_features)
delta_scores = delta_model.decision_function(delta_features)
delta_min_score, delta_max_score = delta_scores.min(), delta_scores.max()

# --- Prediction Function ---
def predict_fraud(entry):
    entry = entry.copy()

    # Predict medication risk
    med_vec = vectorizer.transform([entry["DESCRIPTION_med"]])
    risk_category = risk_classifier.predict(med_vec)[0]
    risk_code = risk_mapping[risk_category]
    entry["MEDICATION_RISK_CODE"] = risk_code

    # Encode using saved label encoders
    for col in label_encoders:
        if col in entry:
            le = label_encoders[col]
            if entry[col] in le.classes_:
                entry[col + "_encoded"] = le.transform([entry[col]])[0]
            else:
                le.classes_ = np.append(le.classes_, entry[col])
                entry[col + "_encoded"] = le.transform([entry[col]])[0]

    # Delta model or general model?
    matched = patient_history[patient_history["PATIENT_ID"] == entry["PATIENT_ID"]]
    if not matched.empty:
        avg = matched.iloc[0]
        delta_base = entry["BASE_COST"] - avg["BASE_COST"]
        delta_cost = entry["TOTALCOST"] - avg["TOTALCOST"]
        delta_disp = entry["DISPENSES"] - avg["DISPENSES"]
        delta_df = pd.DataFrame([{
            "delta_base": delta_base,
            "delta_cost": delta_cost,
            "delta_disp": delta_disp
        }])
        score = delta_model.decision_function(delta_df)[0]
        norm_score = int(np.clip((delta_max_score - score) / (delta_max_score - delta_min_score) * 100, 0, 100))
        model_type = "patient history"
    else:
        entry_df = pd.DataFrame([entry])
        score = general_model.decision_function(entry_df[feature_cols])[0]
        norm_score = int(np.clip((general_max_score - score) / (general_max_score - general_min_score) * 100, 0, 100))
        model_type = "general population"

    is_fraud = score < 0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry.update({
        "fraud": is_fraud,
        "risk_score": norm_score,
        "medication_risk": risk_category,
        "used_model": model_type,
        "timestamp": timestamp
    })

    return entry
