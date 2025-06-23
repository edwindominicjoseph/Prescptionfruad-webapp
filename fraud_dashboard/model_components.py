import os
import random
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# --- Load Dataset ---
DATA_PATH = "C:/Fruad detection_1st_prototype/data/merged_Fullcover.csv"
df3 = pd.read_csv(DATA_PATH)

# --- Medication Risk Lists ---
high_risk_medications = [
    "pregabalin", "gabapentin", "tapentadol", "carfentanil", "nitrazepam", "zopiclone", "zolpidem",
    "lorazepam", "temazepam", "hydromorphone", "fentanyl", "methadone", "oxycodone", "morphine",
    "alfentanil", "hydrocodone bitartrate", "medroxyPROGESTERone acetate", "leuprolide acetate",
    "enoxaparin sodium", "docetaxel", "epinephrine", "fluorouracil", "oxaliplatin", "furosemide",
    "doxorubicin hydrochloride", "fulvestrant", "sufentanil", "abuse-deterrent oxycodone hydrochloride",
    "amiodarone hydrochloride", "vancomycin", "tramadol", "Morphine Sulfate",
    "Oxycodone Hydrochloride", "Codeine Phosphate", "Methadone Hydrochloride",
    "Tramadol Hydrochloride", "Meperidine Hydrochloride", "Buprenorphine / Naloxone",
    "Lorazepam", "Diazepam", "Midazolam", "Clonazepam", "Remifentanil",
    "Nicotine Transdermal Patch", "Propofol"
]
   

moderate_risk_medications = [
    "rivaroxaban", "dabigatran", "azathioprine", "baricitinib", "moxifloxacin", "clarithromycin",
    "erythromycin", "ondansetron", "donepezil hydrochloride", "memantine hydrochloride",
    "metformin hydrochloride", "nicotine transdermal patch", "ethinyl estradiol", "norelgestromin",
    "fluticasone propionate", "liraglutide", "norepinephrine", "alendronic acid", "amoxicillin clavulanate",
    "alprazolam", "salmeterol fluticasone", "piperacillin tazobactam",
    "fentanyl transdermal system", "warfarin", "acetaminophen hydrocodone", "cimetidine",
    "DOCEtaxel", "Epirubicin Hydrochloride", "Cyclophosphamide", "Cisplatin", "Methotrexate",
    "PACLitaxel", "Carboplatin", "Leuprolide Acetate", "Letrozole", "Anastrozole", "Exemestane",
    "Tamoxifen", "Palbociclib", "Ribociclib", "Neratinib", "Lapatinib",
    "Ethinyl Estradiol / Norelgestromin", "Mirena", "Kyleena", "Liletta", "NuvaRing", "Yaz",
    "Levora", "Natazia", "Trinessa", "Camila", "Jolivette", "Errin", "Remdesivir",
    "Heparin sodium porcine", "Alteplase", "Atropine Sulfate", "Desflurane", "Isoflurane",
    "Sevoflurane", "Rocuronium bromide", "Epoetin Alfa", "Glycopyrrolate", "Aviptadil",
    "Leronlimab", "Lenzilumab"
]
risk_mapping = {"Low Risk": 0, "Moderate Risk": 1, "High Risk": 2}

def categorize_risk(med_name: str) -> str:
    name = med_name.lower()
    if any(keyword in name for keyword in [m.lower() for m in high_risk_medications]):
        return "High Risk"
    if any(keyword in name for keyword in [m.lower() for m in moderate_risk_medications]):
        return "Moderate Risk"
    return "Low Risk"

# --- Medication Risk Column ---
df3["MEDICATION_RISK"] = df3["DESCRIPTION_med"].apply(categorize_risk)
df3["MEDICATION_RISK_CODE"] = df3["MEDICATION_RISK"].map(risk_mapping)

# --- Train Medication Risk Classifier ---
vectorizer_new = TfidfVectorizer()
X_vec = vectorizer_new.fit_transform(df3["DESCRIPTION_med"].str.lower())
risk_classifier_new = LogisticRegression(max_iter=1000)
risk_classifier_new.fit(X_vec, df3["MEDICATION_RISK"])

# --- Label Encoding ---
label_encoders_new = {}
encoded_df = df3.copy()
for col in [
    "DESCRIPTION_med", "ENCOUNTERCLASS", "PROVIDER", "ORGANIZATION",
    "GENDER", "ETHNICITY", "MARITAL", "STATE"
]:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(encoded_df[col])
    label_encoders_new[col] = le

encoded_df["MEDICATION_RISK_CODE"] = df3["MEDICATION_RISK"].map(risk_mapping)

# --- General Model ---
feature_cols_new = ["ENCOUNTERCLASS", "DISPENSES", "TOTALCOST", "AGE", "MEDICATION_RISK_CODE"]
general_model_new = IsolationForest(contamination=0.05, random_state=42)
general_model_new.fit(encoded_df[feature_cols_new])
general_scores = general_model_new.decision_function(encoded_df[feature_cols_new])
general_min_score_new, general_max_score_new = general_scores.min(), general_scores.max()

# --- Patient History Model ---
patient_history_new = (
    df3.groupby("PATIENT_med")
    .agg({"BASE_COST": "mean", "TOTALCOST": "mean", "DISPENSES": "mean", "AGE": "first"})
    .reset_index()
    .rename(columns={"PATIENT_med": "PATIENT_ID"})
)

delta_data = df3[["PATIENT_med", "BASE_COST", "TOTALCOST", "DISPENSES"]].merge(
    patient_history_new,
    left_on="PATIENT_med",
    right_on="PATIENT_ID",
    suffixes=("", "_avg")
)
delta_data["delta_base"] = delta_data["BASE_COST"] - delta_data["BASE_COST_avg"]
delta_data["delta_cost"] = delta_data["TOTALCOST"] - delta_data["TOTALCOST_avg"]
delta_data["delta_disp"] = delta_data["DISPENSES"] - delta_data["DISPENSES_avg"]
delta_features = delta_data[["delta_base", "delta_cost", "delta_disp"]]

delta_model_new = IsolationForest(contamination=0.05, random_state=42)
delta_model_new.fit(delta_features)
delta_scores = delta_model_new.decision_function(delta_features)
delta_min_score_new, delta_max_score_new = delta_scores.min(), delta_scores.max()

# --- SHAP Explainability ---
explainer_general = shap.Explainer(general_model_new, encoded_df[feature_cols_new])
explainer_delta = shap.Explainer(delta_model_new, delta_features)

# --- Predict Function ---
def predict_fraud(
    entry: pd.Series | dict,
    vectorizer=vectorizer_new,
    risk_classifier=risk_classifier_new,
    risk_mapping=risk_mapping,
    label_encoders=label_encoders_new,
    general_model=general_model_new,
    delta_model=delta_model_new,
    patient_history=patient_history_new,
    feature_cols=feature_cols_new,
    delta_min=delta_min_score_new,
    delta_max=delta_max_score_new,
    general_min=general_min_score_new,
    general_max=general_max_score_new,
    shap_explainer_general=explainer_general,
    shap_explainer_delta=explainer_delta,
    visualize: bool = False,
) -> dict:
    if isinstance(entry, dict):
        entry = pd.Series(entry)
    entry = entry.copy()

    med_vec = vectorizer.transform([entry["DESCRIPTION_med"].lower()])
    risk_category = risk_classifier.predict(med_vec)[0]
    risk_code = risk_mapping[risk_category]
    entry["MEDICATION_RISK_CODE"] = risk_code

    for col in label_encoders:
        if col in entry:
            le = label_encoders[col]
            if entry[col] in le.classes_:
                entry[col] = le.transform([entry[col]])[0]
            else:
                le.classes_ = np.append(le.classes_, entry[col])
                entry[col] = le.transform([entry[col]])[0]

    entry["PATIENT_ID"] = entry.get("PATIENT_med", entry.get("PATIENT_ID"))
    shap_values = None
    shap_input = None

    matched = patient_history[patient_history["PATIENT_ID"] == entry["PATIENT_ID"]]
    if not matched.empty:
        avg = matched.iloc[0]
        delta_df = pd.DataFrame([{
            "delta_base": entry["BASE_COST"] - avg["BASE_COST"],
            "delta_cost": entry["TOTALCOST"] - avg["TOTALCOST"],
            "delta_disp": entry["DISPENSES"] - avg["DISPENSES"]
        }])
        score = delta_model.decision_function(delta_df)[0]
        norm_score = int(np.clip((delta_max - score) / (delta_max - delta_min) * 100, 0, 100))
        model_type = "patient history"
        if shap_explainer_delta:
            shap_values = shap_explainer_delta(delta_df)
            shap_input = delta_df
    else:
        entry_df = pd.DataFrame([entry])
        score = general_model.decision_function(entry_df[feature_cols])[0]
        norm_score = int(np.clip((general_max - score) / (general_max - general_min) * 100, 0, 100))
        model_type = "general population"
        if shap_explainer_general:
            shap_values = shap_explainer_general(entry_df[feature_cols])
            shap_input = entry_df[feature_cols]

    if visualize and shap_values is not None and shap_input is not None:
        shap.initjs()
        shap.force_plot(
            base_value=shap_values.base_values[0],
            shap_values=shap_values.values[0],
            features=shap_input.iloc[0],
            matplotlib=True,
        )
        plt.show()

    return {
        "fraud": score < 0,
        "risk_score": norm_score,
        "medication_risk": risk_category,
        "used_model": model_type,
        "shap_features": shap_input.to_dict(orient="records")[0] if shap_input is not None else None,
        "shap_values": shap_values.values[0].tolist() if shap_values is not None else None,
    }
