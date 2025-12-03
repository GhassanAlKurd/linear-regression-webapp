import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model artifacts from your script
from linear_regression import (
    load_data,
    scale_features,
    gradient_descent,
    predict,
    r2_score
)

st.set_page_config(page_title="Linear Regression Demo", layout="wide")

st.title("📈 Linear Regression from Scratch (Web App)")
st.write("This app uses gradient descent to fit a line to your CSV data.")

# -------------------------------
# Load Data
# -------------------------------
data = load_data("data_clean.csv")
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]

# Scale
X_scaled, X_mean, X_std = scale_features(X)

# Model Training Controls
st.sidebar.header("Model Controls")
learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.00001,
    max_value=0.01,
    value=0.001,
    step=0.00001,
    format="%.5f"
)

iterations = st.sidebar.slider("Iterations", 100, 5000, 1000, 100)

# Train the Model
b, w, errors, snapshots = gradient_descent(
    X_scaled,
    y,
    learning_rate=learning_rate,
    num_iterations=iterations
)

# Compute Predictions
y_pred = predict(b, w, X_scaled)
score = r2_score(y, y_pred)

# -------------------------------
# Output / Summary
# -------------------------------
st.subheader("Model Summary")
st.write(f"**Intercept (b):** {b:.4f}")
st.write(f"**Slope (w):** {w[0]:.4f}")
st.write(f"**R² Score:** {score:.4f}")

# -------------------------------
# Plot: Regression Fit
# -------------------------------
st.subheader("Regression Line Fit")

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.scatter(X, y, label="Data")
x_vals = np.linspace(min(X), max(X), 200)
x_scaled_vals = ((x_vals - X_mean) / X_std).reshape(-1, 1)
y_vals = predict(b, w, x_scaled_vals)
ax1.plot(x_vals, y_vals, color="red", label="Regression Line")
ax1.legend()
st.pyplot(fig1)

# -------------------------------
# Plot: Error Curve
# -------------------------------
st.subheader("Error Over Iterations")

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(errors)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Mean Squared Error")
st.pyplot(fig2)

# -------------------------------
# User Prediction Tool
# -------------------------------
st.subheader("Prediction Tool")
user_input = st.number_input("Enter an X value:", value=float(X.mean()))
user_scaled_value = float((user_input - X_mean) / X_std)
user_scaled = np.array([[user_scaled_value]])
pred_y = predict(b, w, user_scaled)[0]
st.write(f"### Predicted Y: **{pred_y:.4f}**")
