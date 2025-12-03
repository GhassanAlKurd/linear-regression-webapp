import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model artifacts
from linear_regression import (
    load_data,
    scale_features,
    gradient_descent,
    predict,
    r2_score
)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Linear Regression Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Linear Regression from Scratch (Web App)")
st.write("This app uses gradient descent to fit a line to your CSV data.")

# -------------------------------
# Load Data
# -------------------------------
data = load_data("data_clean.csv")
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]

# Scale data
X_scaled, X_mean, X_std = scale_features(X)

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Model Controls")

learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.00001,
    max_value=0.01,
    value=0.001,
    step=0.00001,
    format="%.5f"
)

iterations = st.sidebar.slider(
    "Iterations",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

st.sidebar.markdown("---")
st.sidebar.write("Use the sliders above to retrain the model.")

# -------------------------------
# Train Model
# -------------------------------
b, w, errors, snapshots = gradient_descent(
    X_scaled,
    y,
    learning_rate=learning_rate,
    num_iterations=iterations
)

# Predictions
y_pred = predict(b, w, X_scaled)
score = r2_score(y, y_pred)

# -------------------------------
# Model Summary
# -------------------------------
st.subheader("📊 Model Summary")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Intercept (b)", f"{b:.4f}")
with col2:
    st.metric("Slope (w)", f"{w[0]:.4f}")
with col3:
    st.metric("R² Score", f"{score:.4f}")

st.markdown("---")

# -------------------------------
# Regression Line Plot
# -------------------------------
st.subheader("📉 Regression Line Fit")

fig1, ax1 = plt.subplots(figsize=(5.5, 3.5), dpi=120)
ax1.scatter(X, y, label="Data", s=20)

x_vals = np.linspace(min(X), max(X), 200)
x_scaled_vals = ((x_vals - X_mean) / X_std).reshape(-1, 1)
y_vals = predict(b, w, x_scaled_vals)

ax1.plot(x_vals, y_vals, color="red", label="Regression Line", linewidth=2)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()

st.pyplot(fig1, clear_figure=True)

st.markdown("---")

# -------------------------------
# Error Curve Plot
# -------------------------------
st.subheader("📉 Error Over Iterations")

fig2, ax2 = plt.subplots(figsize=(5.5, 3.5), dpi=120)
ax2.plot(errors, linewidth=2)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Mean Squared Error")

st.pyplot(fig2, clear_figure=True)

st.markdown("---")

# -------------------------------
# Prediction Tool
# -------------------------------
st.subheader("🔮 Prediction Tool")

user_input = st.number_input(
    "Enter an X value:",
    value=float(X.mean())
)

user_scaled_value = float((user_input - X_mean) / X_std)
user_scaled = np.array([[user_scaled_value]])
pred_y = predict(b, w, user_scaled)[0]

st.write(f"### Predicted Y: **{pred_y:.4f}**")
