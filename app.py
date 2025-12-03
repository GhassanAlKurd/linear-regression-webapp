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

st.markdown(
    """
    <style>
        .main {
            background-color: #f4f6fb;
            font-family: "Inter", "Segoe UI", system-ui;
        }
        .metric-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0px 10px 25px rgba(15, 23, 42, 0.06);
        }
        .section-card {
            border-radius: 18px;
            padding: 1.5rem;
            background: linear-gradient(135deg, #fefefe 0%, #f2f8ff 100%);
            border: 1px solid #e2e8f0;
        }
        .prediction-card {
            border-radius: 18px;
            padding: 1.5rem;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            box-shadow: 0px 12px 28px rgba(15, 23, 42, 0.08);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Linear Regression from Scratch (Web App)")
st.write("This app uses gradient descent to fit a line to your CSV data.")

# -------------------------------
# Load Data
# -------------------------------
data = load_data("data_clean.csv")
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]
data_df = pd.DataFrame(data, columns=["X", "Y"])

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


def metric_card(title: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <p style="margin:0;color:#64748b;font-size:0.85rem;">{title}</p>
            <p style="margin:0;color:#0f172a;font-size:1.6rem;font-weight:600;">{value}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------
# Model Summary
# -------------------------------
st.subheader("📊 Model Summary")

summary_cols = st.columns(3, gap="large")
with summary_cols[0]:
    metric_card("Intercept (b)", f"{b:.4f}")
with summary_cols[1]:
    metric_card("Slope (w)", f"{w[0]:.4f}")
with summary_cols[2]:
    metric_card("R² Score", f"{score:.4f}")

# Dataset stats
dataset_cols = st.columns(3, gap="large")
with dataset_cols[0]:
    metric_card("Samples", f"{len(X):,}")
with dataset_cols[1]:
    metric_card("Avg X", f"{X.mean():.2f}")
with dataset_cols[2]:
    metric_card("Avg Y", f"{y.mean():.2f}")

with st.expander("Peek at dataset"):
    st.dataframe(data_df.head(10), use_container_width=True)

st.divider()

# -------------------------------
# Regression Line Plot
# -------------------------------
st.subheader("📉 Visual Insights")
tab_fit, tab_error = st.tabs(["Regression Fit", "Error Curve"])

with tab_fit:
    with st.container():
        fig1, ax1 = plt.subplots(figsize=(3.4, 2.1), dpi=180)
        ax1.scatter(X, y, label="Data", s=16, color="#6366f1", alpha=0.75)
        x_vals = np.linspace(min(X), max(X), 200)
        x_scaled_vals = ((x_vals - X_mean) / X_std).reshape(-1, 1)
        y_vals = predict(b, w, x_scaled_vals)
        ax1.plot(x_vals, y_vals, color="#ef4444", label="Regression Line", linewidth=2)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.legend()
        ax1.grid(alpha=0.25)
        fig1.tight_layout()
        st.pyplot(fig1, clear_figure=True, use_container_width=False)

with tab_error:
    with st.container():
        fig2, ax2 = plt.subplots(figsize=(3.4, 2.1), dpi=180)
        ax2.plot(errors, linewidth=2, color="#0ea5e9")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Mean Squared Error")
        ax2.grid(alpha=0.4, linestyle="--", linewidth=0.6)
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True, use_container_width=False)

st.divider()

# -------------------------------
# Prediction Tool
# -------------------------------
st.subheader("🔮 Prediction Tool")

st.caption("Explore the fitted line by selecting any X value.")
with st.container():
    slider = st.slider(
        "Choose an X value",
        float(X.min()),
        float(X.max()),
        float(X.mean())
    )

    user_scaled_value = float((slider - X_mean) / X_std)
    user_scaled = np.array([[user_scaled_value]])
    pred_y = predict(b, w, user_scaled)[0]

    st.markdown(
        f"""
        <div class="prediction-card">
            <p style="margin:0;color:#6366f1;font-size:0.9rem;">Predicted Y</p>
            <p style="margin:0;color:#0f172a;font-size:2rem;font-weight:700;">{pred_y:.4f}</p>
            <p style="margin:0;color:#94a3b8;font-size:0.85rem;">For X = {slider:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
