import streamlit as st
import pandas as pd
import joblib

# ---------- CONFIG ----------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="wide")

# ---------- LOAD ----------
model = joblib.load("LR_HeartDisease.pkl")
scaler = joblib.load("scaler.pkl")
cols = joblib.load("columns.pkl")

# ---------- SIMPLE DARK CSS ----------
st.markdown("""
<style>
.stApp {background:#0f172a; color:white;}
h1 {text-align:center; color:#38bdf8;}

.stButton>button {
    width:100%; background:#38bdf8; color:black;
    border-radius:8px; height:2.5em;
}

.high {border-left:5px solid #ef4444; padding:10px; background:#1e293b;}
.low {border-left:5px solid #22c55e; padding:10px; background:#1e293b;}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1>🫀 Heart Disease Predictor</h1>", unsafe_allow_html=True)

# ---------- INPUT ----------
c1, c2, c3 = st.columns(3)

with c1:
    age = st.slider("Age",18,100,40)
    sex = st.selectbox("Sex",["M","F"])
    cp = st.selectbox("Chest Pain",["ATA","NAP","TA","ASY"])
    fbs = st.selectbox("Fasting BS",[0,1])

with c2:
    bp = st.number_input("Rest BP",80,200,120)
    chol = st.number_input("Cholesterol",100,600,200)
    ecg = st.selectbox("ECG",["Normal","ST","LVH"])
    slope = st.selectbox("ST Slope",["Up","Flat","Down"])

with c3:
    hr = st.slider("Max HR",60,220,150)
    angina = st.selectbox("Ex Angina",["Y","N"])
    oldpeak = st.slider("Oldpeak",0.0,6.0,1.0)

# ---------- PREDICT ----------
if st.button("🔍 Predict"):

    data = {
        'Age': age,
        'RestingBP': bp,
        'Cholesterol': chol,
        'FastingBS': fbs,
        'MaxHR': hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + cp: 1,
        'RestingECG_' + ecg: 1,
        'ExerciseAngina_' + angina: 1,
        'ST_Slope_' + slope: 1
    }

    df = pd.DataFrame([data])

    for col in cols:
        if col not in df:
            df[col] = 0

    df = df[cols]
    df = scaler.transform(df)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    # ---------- RESULT ----------
    r1, r2, r3 = st.columns(3)
    r1.metric("Risk %", f"{prob*100:.1f}")
    r2.metric("Max HR", hr)
    r3.metric("Cholesterol", chol)

    st.progress(prob)

    if pred == 1:
        st.markdown(f"<div class='high'>⚠️ High Risk ({prob*100:.1f}%)</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='low'>✅ Low Risk ({(1-prob)*100:.1f}%)</div>", unsafe_allow_html=True)