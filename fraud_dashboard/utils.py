import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def plot_fraud_pie(df_recent):
    st.subheader("🧠 Fraud Distribution")
    counts = df_recent["fraud"].value_counts()
    labels = ["Fraud", "Not Fraud"]
    values = [counts.get(True, 0), counts.get(False, 0)]
    colors = ["#e74c3c", "#2ecc71"]
    if sum(values) == 0:
        st.info("No data to display in pie chart.")
        return
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    ax.axis("equal")
    st.pyplot(fig)


def plot_top_meds_donut(df_recent, top_n=5):
    st.subheader("🍩 Top Dispensed Medicines")

    top_meds = df_recent["DESCRIPTION_med"].value_counts().head(top_n)
    values = top_meds.values
    meds = [str(med)[:40] + "..." if len(str(med)) > 43 else str(med) for med in top_meds.index]

    colors = plt.cm.tab10.colors[:len(meds)]

    # ✅ Set fixed figure size to match other charts (e.g., 6x4)
    fig, ax = plt.subplots(figsize=(6, 4))  # You can tune this to match exactly
    wedges, _, autotexts = ax.pie(
        values,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.3),
        colors=colors
    )

    ax.axis("equal")
    ax.legend(wedges, meds, title="Medication", loc="center left", bbox_to_anchor=(1, 0.5))

    st.pyplot(fig)





def plot_risk_score_trend(df_recent):
    st.subheader("📈  Real-Time Risk Score Trend")

    df_recent["timestamp"] = pd.to_datetime(df_recent["timestamp"], errors="coerce")
    df_sorted = df_recent.sort_values("timestamp")

    if df_sorted.empty:
        st.info("No risk score data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 2.5))  # 👈 Wide and short
    ax.plot(df_sorted["timestamp"], df_sorted["risk_score"], color='red', linewidth=2)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Risk Score")
    ax.set_title("Risk Score Trend (Last Entries)", fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)

    st.pyplot(fig)


