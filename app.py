# ==========================================
# TITANIC SURVIVAL PREDICTION SYSTEM
# STREAMLIT DEPLOYMENT
# ==========================================

# ------------------------------------------
# IMPORT LIBRARIES
# ------------------------------------------

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px

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
# EVALUATION METRICS
# ------------------------------------------

y_pred_prob = model.predict(X)

y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y, y_pred)

precision = precision_score(y, y_pred)

recall = recall_score(y, y_pred)

f1 = f1_score(y, y_pred)

r2 = r2_score(y, y_pred)

# ------------------------------------------
# HEADER SECTION
# ------------------------------------------

st.markdown("""
# 🚢 Titanic Survival Prediction System

### Deep Learning Based Passenger Survival Prediction
""")

st.image(
    "https://cdn-icons-png.flaticon.com/512/2972/2972285.png",
    width=120
)

st.divider()

# ------------------------------------------
# PROJECT DESCRIPTION
# ------------------------------------------

st.markdown("""
## 📌 Project Description

This application predicts whether a passenger
would survive or not during the Titanic disaster
using an Artificial Neural Network (ANN).

The model is trained using TensorFlow/Keras
and deployed using Streamlit Community Cloud.
""")

st.divider()

# ------------------------------------------
# PASSENGER INPUT FORM
# ------------------------------------------

st.markdown("## 🧾 Passenger Input Form")

col1, col2, col3 = st.columns(3)

with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:

    age = st.slider(
        "Age",
        min_value=1,
        max_value=80,
        value=24
    )

with col3:

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=120.0
    )

st.divider()

# ------------------------------------------
# PREPROCESSING
# ------------------------------------------

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

# ------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------

if st.button("🔍 Predict Survival"):

    prediction = model.predict(input_data)

    probability = prediction[0][0]

    st.divider()

    # --------------------------------------
    # RESULT SECTION
    # --------------------------------------

    st.markdown("## 🎯 Prediction Result")

    if probability > 0.5:

        result = "Survived"

        st.success(
            "Passenger Likely Survived"
        )

    else:

        result = "Not Survived"

        st.error(
            "Passenger Likely Did Not Survive"
        )

    # --------------------------------------
    # METRICS
    # --------------------------------------

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
            f"{max(probability, 1-probability)*100:.2f}%"
        )

    st.divider()

    # --------------------------------------
    # PIE CHART
    # --------------------------------------

    st.markdown(
        "## 📊 Survival Probability Visualization"
    )

    labels = [
        "Survived",
        "Not Survived"
    ]

    values = [
        probability,
        1 - probability
    ]

    fig = px.pie(
        names=labels,
        values=values,
        hole=0.4,
        title="Passenger Survival Probability"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------
    # MODEL EVALUATION METRICS
    # --------------------------------------

    st.markdown(
        "## 📈 Model Evaluation Metrics"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy*100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{precision*100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{recall*100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{f1:.2f}"
        )

    with col5:

        st.metric(
            "R2 Score",
            f"{r2:.2f}"
        )