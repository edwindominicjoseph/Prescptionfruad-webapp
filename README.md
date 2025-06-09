# 💊 AI-Powered Prescription Fraud Detection Platform (Alpha)

This project is a predictive analytics and machine learning platform designed to detect fraudulent prescriptions and assess medication risk using natural language processing and anomaly detection.

🚀 **Test Version Now Live**:  
🔗 [Try it here](https://app-fraud-detection-jqbudfc99zxkaaqku7yhvx.streamlit.app/)

---

## 🚀 Features

- ✅ **Medication Risk Classification** using TF-IDF + Logistic Regression  
- ✅ **Fraud Detection** using Isolation Forest (general + patient-history based)  
- ✅ **Dynamic Risk Scoring** combining model outputs  
- ✅ **Streamlit Dashboard** with interactive charts, metrics, and visual alerts  
- ✅ **Modular & Scalable Codebase** for easy updates and deployment  
- 🔜 **SHAP Explainability** and **CI/CD + Docker support** coming soon  

---

## 🧠 Predictive Modeling

### 📌 Medication Risk Classification
- **Method**: TF-IDF Vectorizer + Logistic Regression  
- **Goal**: Categorize new medications as **Low**, **Medium**, or **High Risk**  
- **Input**: Medication name or description  
- **Fallback**: Uses predefined category list if available  

### 📌 Fraud Detection
- **General Model**: Detects anomalies in overall prescription patterns  
- **Patient-History Model**: Flags prescriptions deviating from a patient’s medication history  
- **Output**: Combined fraud risk score between 0 (safe) and 1 (suspicious)  

---

## 📊 Streamlit Dashboard

- Real-time fraud scoring  
- Medication risk overview  
- Top 10 risky prescriptions  
- Flagged cases table  
- Lightweight, wide-layout UI  

▶️ **Run locally**:
```bash
streamlit run app.py
