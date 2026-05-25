# ==========================================
# TITANIC SURVIVAL PREDICTION SYSTEM
# CREATIVE STREAMLIT DEPLOYMENT
# ==========================================

# ------------------------------------------
# IMPORT LIBRARIES
# ------------------------------------------

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score
)

# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ------------------------------------------
# LOAD MODEL
# ------------------------------------------

model = tf.keras.models.load_model("model.h5")

# ------------------------------------------
# LOAD DATASET
# ------------------------------------------

df = pd.read_csv("Titanic-Dataset.csv")

data = df[['Pclass', 'Age', 'Fare', 'Survived']]

# ------------------------------------------
# HANDLE MISSING VALUES
# ------------------------------------------

data['Age'].fillna(
    data['Age'].mean(),
    inplace=True
)

# ------------------------------------------
# NORMALIZATION
# ------------------------------------------

data['Pclass'] = (
    (data['Pclass'] - data['Pclass'].min())
    /
    (data['Pclass'].max() - data['Pclass'].min())
)

data['Age'] = (
    (data['Age'] - data['Age'].min())
    /
    (data['Age'].max() - data['Age'].min())
)

data['Fare'] = (
    (data['Fare'] - data['Fare'].min())
    /
    (data['Fare'].max() - data['Fare'].min())
)

# ------------------------------------------
# INPUTS AND OUTPUT
# ------------------------------------------

X = data[['Pclass', 'Age', 'Fare']]

y = data['Survived']

# ------------------------------------------
# MODEL EVALUATION METRICS
# ------------------------------------------

y_pred_prob = model.predict(X)

y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y, y_pred)

precision = precision_score(y, y_pred)

recall = recall_score(y, y_pred)

f1 = f1_score(y, y_pred)

r2 = r2_score(y, y_pred)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
    margin-bottom: 25px;
}

.metric-card {
    background: rgba(255,255,255,0.07);
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(255,255,255,0.1);
}

h1,h2,h3,h4 {
    color: white;
}

p,label {
    color: white !important;
}

.stButton>button {
    width: 100%;
    height: 60px;
    border-radius: 15px;
    border: none;
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    font-size: 20px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(to right, #0072ff, #00c6ff);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER SECTION
# ==========================================

st.markdown("""
<div class="card">

# 🚢 Titanic Survival Prediction System

### Deep Learning Based Passenger Survival Prediction

AI Powered Passenger Risk Analysis using Artificial Neural Networks

</div>
""", unsafe_allow_html=True)

# ==========================================
# PROJECT DESCRIPTION
# ==========================================

st.markdown("""
<div class="card">

## 📌 Project Description

This AI-powered application predicts whether a passenger
would survive during the Titanic disaster using
Artificial Neural Networks (ANN).

### Technologies Used:
- TensorFlow / Keras
- Deep Learning
- Streamlit Deployment
- ANN Classification

### Features Used:
- Passenger Class
- Age
- Fare

</div>
""", unsafe_allow_html=True)

# ==========================================
# INPUT SECTION
# ==========================================

st.markdown("""
<div class="card">

## 🧾 Passenger Input Form

</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# ------------------------------------------
# PASSENGER CLASS
# ------------------------------------------

with col1:

    st.markdown("###  Passenger Class")

    pclass_option = st.selectbox(
        "",
        [
            "First Class",
            "Second Class",
            "Third Class"
        ]
    )

    if pclass_option == "First Class":
        pclass = 1

    elif pclass_option == "Second Class":
        pclass = 2

    else:
        pclass = 3

# ------------------------------------------
# AGE
# ------------------------------------------

with col2:

    st.markdown("###  Age")

    age = st.slider(
        "",
        min_value=1,
        max_value=80,
        value=24
    )

# ------------------------------------------
# FARE
# ------------------------------------------

with col3:

    st.markdown("###  Fare")

    fare = st.number_input(
        "",
        min_value=0.0,
        value=120.0
    )

# ==========================================
# PREPROCESSING
# ==========================================

pclass_norm = (pclass - 1) / (3 - 1)

age_norm = (age - 0.42) / (80 - 0.42)

fare_norm = (fare - 0) / (512 - 0)

input_data = np.array([
    [
        pclass_norm,
        age_norm,
        fare_norm
    ]
])

# ==========================================
# PREDICTION BUTTON
# ==========================================

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Survival"):

    prediction = model.predict(input_data)

    probability = prediction[0][0]

    if probability > 0.5:
        result = "Survived"
        status = "✅ Passenger Likely Survived"
    else:
        result = "Not Survived"
        status = "❌ Passenger Likely Did Not Survive"

    # ======================================
    # RESULT SECTION
    # ======================================

    st.markdown("""
    <div class="card">

    ## 🎯 Prediction Result

    </div>
    """, unsafe_allow_html=True)

    st.success(status)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Prediction",
            result
        )

    with col2:
        st.metric(
            "Survival Probability",
            f"{probability*100:.2f}%"
        )

    with col3:
        st.metric(
            "Confidence Score",
            f"{max(probability,1-probability)*100:.2f}%"
        )

    # ======================================
    # SPEEDOMETER GAUGE
    # ======================================

    st.markdown("""
    <div class="card">

    ## 🚦 Survival Probability Meter

    </div>
    """, unsafe_allow_html=True)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Survival Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "cyan"},
            'steps': [
                {'range': [0, 50], 'color': "#ff4b4b"},
                {'range': [50, 100], 'color': "#00ff99"}
            ]
        }
    ))

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # ======================================
    # PIE CHART
    # ======================================

    st.markdown("""
    <div class="card">

    ## 🥧 Probability Distribution

    </div>
    """, unsafe_allow_html=True)

    labels = [
        "Survived",
        "Not Survived"
    ]

    values = [
        probability,
        1 - probability
    ]

    pie_chart = px.pie(
        names=labels,
        values=values,
        hole=0.5,
        title="Survival vs Non-Survival"
    )

    st.plotly_chart(
        pie_chart,
        use_container_width=True
    )

    # ======================================
    # CREATIVE EVALUATION METRICS
    # ======================================

    st.markdown("""
    <div class="card">

    ## 📈 Model Performance Dashboard

    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "🎯 Accuracy",
            f"{accuracy*100:.2f}%"
        )

    with c2:
        st.metric(
            "📌 Precision",
            f"{precision*100:.2f}%"
        )

    with c3:
        st.metric(
            "🔁 Recall",
            f"{recall*100:.2f}%"
        )

    with c4:
        st.metric(
            "⚖️ F1 Score",
            f"{f1:.2f}"
        )

    with c5:
        st.metric(
            "📊 R2 Score",
            f"{r2:.2f}"
        )