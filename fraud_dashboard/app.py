import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
from model_components import predict_fraud
from utils import (
    plot_fraud_pie,
    plot_risk_score_trend,
    plot_top_meds_donut
)

# --- Page Setup ---
st.set_page_config(page_title="Prescription Fraud Dashboard", layout="wide")
st.title("💊 Prescription Fraud Detection")
import os
import pandas as pd






# --- Load Predictions File ---
PREDICTIONS_PATH = "outputs/predictions.csv"
os.makedirs("outputs", exist_ok=True)

if os.path.exists(PREDICTIONS_PATH) and os.path.getsize(PREDICTIONS_PATH) > 0:
    df_results = pd.read_csv(PREDICTIONS_PATH)
    df_results["timestamp"] = pd.to_datetime(df_results["timestamp"], errors="coerce")
else:
    df_results = pd.DataFrame()

# --- Summary Metrics ---
st.markdown("## 📊 Fraud Detection Overview")
col1, col2, col3 = st.columns(3)
total_frauds = df_results["fraud"].sum() if not df_results.empty else 0
monthly_frauds = df_results[df_results["timestamp"] >= datetime.now() - timedelta(days=30)]["fraud"].sum() if not df_results.empty else 0
detection_rate = round((total_frauds / len(df_results)) * 100, 1) if len(df_results) else 0
col1.metric("Total Fraud Cases", f"{int(total_frauds)}", "▲ 5%")
col2.metric("Monthly Fraud Increase", f"{int(monthly_frauds)}", "▲ 2.5%")
col3.metric("Detection Rate", f"{detection_rate}%", "-")

# --- Single Entry Form ---
with st.expander("🔍 Check Single Prescription"):
    with st.form("entry_form"):
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("Patient ID", "demo_patient_01")
            med = st.text_input("Medication Name", "Oxycodone Hydrochloride 10 MG")
            enc = st.selectbox("Encounter Type", ["emergency", "inpatient", "ambulatory"])
            disp = st.number_input("Dispenses", 1, 10, value=2)
            base = st.number_input("Base Cost", 0.0, 500.0, value=20.0)
            cost = st.number_input("Total Cost", 0.0, 5000.0, value=500.0)
        with col2:
            age = st.slider("Age", 0, 100, 45)
            gender = st.selectbox("Gender", ["M", "F"])
            ethnicity = st.selectbox("Ethnicity", ["white", "black", "asian", "hispanic"])
            marital = st.selectbox("Marital Status", ["M", "S", "D", "W"])
            state = st.text_input("State", "Massachusetts")
            provider = st.text_input("Provider", "Dr.ABC")
            org = st.text_input("Organization", "City Health")
        submit = st.form_submit_button("🚨 Check Fraud")

    if submit:
        entry = {
            "PATIENT_ID": patient_id, "DESCRIPTION_med": med, "ENCOUNTERCLASS": enc,
            "DISPENSES": disp, "BASE_COST": base, "TOTALCOST": cost, "AGE": age,
            "GENDER": gender, "ETHNICITY": ethnicity, "MARITAL": marital,
            "STATE": state, "PROVIDER": provider, "ORGANIZATION": org
        }
        result = predict_fraud(entry)
        result["timestamp"] = datetime.now()
        df_results = pd.concat([df_results, pd.DataFrame([result])], ignore_index=True)
        df_results.to_csv(PREDICTIONS_PATH, index=False)

        st.subheader("🧠 Prediction Result")
        st.metric("Risk Score", f"{result['risk_score']} / 100")
        st.write(f"Medication Risk: **{result['medication_risk']}**")
        st.caption(f"Used Model: {result['used_model']}")
        st.error("🟥 Fraud Detected" if result["fraud"] else "🟩 Not Fraudulent")

# --- Top Visualizations ---
st.markdown("### 📊 Visual Insights (Last 7 Days)")
if not df_results.empty:
    df_recent = df_results.copy()
    df_recent["timestamp"] = pd.to_datetime(df_recent["timestamp"], errors="coerce")
    df_recent = df_recent[df_recent["timestamp"].notnull() & (df_recent["timestamp"] >= datetime.now() - timedelta(days=7))]



# Narrow cards row (2 cols)
col1, col2 = st.columns(2)
with col1:
    plot_fraud_pie(df_recent)
with col2:
    plot_top_meds_donut(df_recent)

# Wide full-row risk trend
plot_risk_score_trend(df_recent)  # Full width here



    


# --- Flagged Prescriptions Table ---
st.markdown("### 🚨 Flagged Patients")
flagged = df_results[df_results["fraud"] == True].copy()
if not flagged.empty:
    flagged = flagged.tail(10)
    flagged["Fraud Risk"] = flagged["risk_score"].apply(lambda x: "⭐" * round(x / 20))
    flagged["Status"] = flagged["fraud"].map({True: "Flagged", False: "Cleared"})
    flagged["Flag Icon"] = flagged["risk_score"].apply(lambda x: "❗ High" if x > 75 else "⚠️ Med" if x > 50 else "✅ Low")
    flagged["Doctor"] = flagged["PROVIDER"]

    for i, row in flagged.iterrows():
        with st.container():
            cols = st.columns([1, 2, 2, 2, 2, 1, 2])
            cols[0].write(f"**{i+1:03}**")
            cols[1].write(row["PATIENT_ID"])
            cols[2].write(row["Status"])
            cols[3].write(row["Fraud Risk"])
            cols[4].write(row["Flag Icon"])
            cols[5].write("👨‍⚕️")
            cols[6].button("🔍 Review", key=f"review_{i}")
else:
    st.info("No flagged entries yet.")


# --- Clear Predictions ---


PREDICTIONS_PATH = "outputs/predictions.csv"
st.markdown("### 🧹 Clear Predictions")
if st.button("Clear All Results"):
    # Overwrite with just headers if file exists
    if os.path.exists(PREDICTIONS_PATH):
        df_empty = pd.DataFrame(columns=[
            "PATIENT_ID", "DESCRIPTION_med", "ENCOUNTERCLASS", "DISPENSES",
            "BASE_COST", "TOTALCOST", "AGE", "GENDER", "ETHNICITY", "MARITAL",
            "STATE", "PROVIDER", "ORGANIZATION",
            "fraud", "risk_score", "medication_risk", "used_model", "timestamp"
        ])
        df_empty.to_csv(PREDICTIONS_PATH, index=False)
        st.success("✅ All predictions have been cleared.")
        
    else:
        st.info("No predictions found to clear.")


