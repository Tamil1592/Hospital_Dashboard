# ============================================
# ER COMMAND HOSPITAL CENTER
# AIIMS + ML + FASTAPI + GPS SIMULATION
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ============================================
# CONFIG
# ============================================

st.set_page_config(
    page_title="ER Command Hospital Center",
    layout="wide",
    page_icon="🏥"
)

st_autorefresh(interval=10000, key="refresh_v4")

# ============================================
# THEME
# ============================================

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
    background:#070b18;
    color:white;
}
[data-testid="stSidebar"]{
    background:#020617;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA SETUP
# ============================================

states = ["TN","KL","KA","MH","DL","UP","WB","GJ","RJ","AP"]
hospitals = ["AIIMS","Apollo","Fortis","Gov Hospital","Metro"]
departments = ["Emergency","Cardiology","Neurology","ICU","Trauma"]

severity_map = {"Critical":3,"Urgent":2,"Moderate":1,"Stable":0}

# ============================================
# DATA GENERATION
# ============================================

def generate_data(n=800):
    data = []
    for i in range(n):
        severity = random.choice(list(severity_map.keys()))
        icu = random.choice([0, 1])
        wait = random.randint(5, 180)
        doctor_load = random.randint(1, 30)

        data.append({
            "State": random.choice(states),
            "Hospital": random.choice(hospitals),
            "Department": random.choice(departments),
            "Severity": severity,
            "SeverityScore": severity_map[severity],
            "ICU": icu,
            "WaitingTime": wait,
            "DoctorLoad": doctor_load,
            "BedsUsed": random.randint(50, 400),
            "Lat": random.uniform(8, 34),
            "Lon": random.uniform(70, 90)
        })
    return pd.DataFrame(data)

df = generate_data()

# ============================================
# ML MODEL (ICU PREDICTION)
# ============================================

X = df[["SeverityScore", "WaitingTime", "DoctorLoad", "BedsUsed"]]
y = df["ICU"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

df["ICU_PRED"] = model.predict(X)

# ============================================
# RISK SCORE ENGINE
# ============================================

df["RiskScore"] = (
    df["SeverityScore"] * 25 +
    df["WaitingTime"] * 0.4 +
    df["DoctorLoad"] * 2
)

# ============================================
# SIDEBAR FILTERS
# ============================================

st.sidebar.title("⚙️ CONTROL CENTER")

state_f = st.sidebar.multiselect("State", df["State"].unique(), df["State"].unique())
hospital_f = st.sidebar.multiselect("Hospital", df["Hospital"].unique(), df["Hospital"].unique())
dept_f = st.sidebar.multiselect("Department", df["Department"].unique(), df["Department"].unique())

filtered = df[
    (df["State"].isin(state_f)) &
    (df["Hospital"].isin(hospital_f)) &
    (df["Department"].isin(dept_f))
]

# ============================================
# KPI DASHBOARD
# ============================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Patients", len(filtered))
col2.metric("ICU Cases", filtered["ICU"].sum())
col3.metric("Predicted ICU", filtered["ICU_PRED"].sum())
col4.metric("Avg Risk", round(filtered["RiskScore"].mean(), 2))

# ============================================
# AI RISK CHART
# ============================================

st.subheader("🧠 AI Patient Risk Scoring")

fig_risk = px.histogram(filtered, x="RiskScore", nbins=30, title="Risk Distribution")
st.plotly_chart(fig_risk, use_container_width=True)

# ============================================
# ICU PREDICTION
# ============================================

st.subheader("🏥 ICU Prediction Model")

fig_icu = px.bar(
    filtered.groupby("Department")["ICU_PRED"].sum().reset_index(),
    x="Department",
    y="ICU_PRED",
    title="Predicted ICU Demand"
)

st.plotly_chart(fig_icu, use_container_width=True)

# ============================================
# PATIENT FLOW
# ============================================

c1, c2 = st.columns(2)

with c1:
    st.subheader("Severity Distribution")
    st.plotly_chart(px.pie(filtered, names="Severity"), use_container_width=True)

with c2:
    st.subheader("ICU Prediction Count")
    st.plotly_chart(px.bar(filtered["ICU_PRED"].value_counts()), use_container_width=True)

# ============================================
# 🚑 LIVE GPS MAP
# ============================================

st.subheader("🚑 Live Ambulance GPS Tracking")

ambulance = filtered.sample(min(80, len(filtered)))

fig_map = px.scatter_mapbox(
    ambulance,
    lat="Lat",
    lon="Lon",
    color="RiskScore",
    size="DoctorLoad",
    zoom=3,
    height=500,
    hover_name="Hospital"
)

fig_map.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig_map, use_container_width=True)

# ============================================
# FASTAPI STATUS
# ============================================

st.subheader("🌐 FastAPI Backend Status")

api_status = random.choice(["ONLINE", "DEGRADED", "OFFLINE"])

if api_status == "ONLINE":
    st.success("FastAPI Connected - Real-Time Data Flow Active")
elif api_status == "DEGRADED":
    st.warning("FastAPI Slow Response")
else:
    st.error("FastAPI Disconnected")

# ============================================
# COMPARISON
# ============================================

st.subheader("🏥 Hospital Intelligence Comparison")

comp = filtered.groupby("Hospital").agg({
    "RiskScore": "mean",
    "WaitingTime": "mean",
    "ICU": "sum"
}).reset_index()

st.plotly_chart(
    px.bar(comp, x="Hospital", y="RiskScore", title="Hospital Risk Score"),
    use_container_width=True
)

# ============================================
# ALERT SYSTEM
# ============================================

st.subheader("⚠️ Emergency AI Alerts")

if filtered["RiskScore"].mean() > 120:
    st.error("🚨 HIGH SYSTEM RISK DETECTED")

if filtered["ICU_PRED"].sum() > 200:
    st.warning("⚠️ ICU SURGE EXPECTED")

if filtered["WaitingTime"].mean() > 90:
    st.warning("⚠️ QUEUE OVERFLOW RISK")

# ============================================
# LIVE TABLE
# ============================================

st.subheader("📄 Live Patient Intelligence")

st.dataframe(filtered.sort_values("RiskScore", ascending=False))

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.caption("ER Command Hospital Center | AI + ML + FASTAPI + GPS SYSTEM")
