# ==========================================
# TITANIC SURVIVAL PREDICTION SYSTEM
# ANN MODEL TRAINING
# ==========================================

# ------------------------------------------
# IMPORT LIBRARIES
# ------------------------------------------

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    confusion_matrix
)

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

# ------------------------------------------
# LOAD DATASET
# ------------------------------------------

df = pd.read_csv("Titanic-Dataset.csv")

# ------------------------------------------
# SELECT REQUIRED COLUMNS
# ------------------------------------------

data = df[['Pclass', 'Age', 'Fare', 'Survived']]

# ------------------------------------------
# HANDLE MISSING VALUES
# ------------------------------------------

data['Age'].fillna(
    data['Age'].mean(),
    inplace=True
)

# ------------------------------------------
# INPUTS AND OUTPUT
# ------------------------------------------

X = data[['Pclass', 'Age', 'Fare']]

y = data['Survived']

# ------------------------------------------
# NORMALIZATION
# ------------------------------------------

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# ------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------
# BUILD ANN MODEL
# ------------------------------------------

model = Sequential()

# Hidden Layer
model.add(
    Dense(
        2,
        input_dim=3,
        activation='sigmoid'
    )
)

# Output Layer
model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)

# ------------------------------------------
# COMPILE MODEL
# ------------------------------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ------------------------------------------
# TRAIN MODEL
# ------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=10
)

# ------------------------------------------
# MODEL EVALUATION
# ------------------------------------------

y_pred_prob = model.predict(X_test)

y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

# ------------------------------------------
# PRINT METRICS
# ------------------------------------------

print("\n========== MODEL EVALUATION ==========\n")

print("Accuracy Score :", round(accuracy,4))

print("Precision Score :", round(precision,4))

print("Recall Score :", round(recall,4))

print("F1 Score :", round(f1,4))

print("R2 Score :", round(r2,4))

print("\nConfusion Matrix:\n")

print(cm)

# ------------------------------------------
# SAVE MODEL
# ------------------------------------------

model.save("model.h5")

print("\nMODEL SAVED SUCCESSFULLY")