import matplotlib

# Use a non-interactive backend so the module works inside Streamlit and headless envs.
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ========================
#   VISUALIZATION
# ========================

def plot_correlation_heatmap(points):
    df = pd.DataFrame(points, columns=["X", "Y"])
    plt.figure(figsize=(6,4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation")
    plt.show()


def plot_regression(points, b, w):
    """
    Plot raw data vs. regression line.
    Works only for one feature (which we currently use).
    """
    x = points[:, 0]
    y = points[:, 1]
    y_pred = b + w * x

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="Data points")
    plt.plot(x, y_pred, color="red", label="Best fit line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression: Best Fit Line")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error(errors):
    plt.figure(figsize=(8, 5))
    plt.plot(errors)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    plt.title("Error Reduction Over Time")
    plt.tight_layout()
    plt.show()


def plot_residuals(y, y_pred):
    residuals = y - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red')
    plt.xlabel("Predicted Y")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals Plot")
    plt.tight_layout()
    plt.show()


# ========================
#   CORE MATH / MODEL
# ========================

def scale_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def load_data(path: str) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",")


def r2_score(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)


def predict(b, w, X):
    X = np.asarray(X)
    return b + np.dot(X, w)


def gradient_descent(
    X, y,
    learning_rate,
    num_iterations,
    initial_b=0.0,
    initial_w=None,
    record_steps=20
):
    """
    Multivariate gradient descent with snapshot recording for animation.
    Saves a snapshot every 'record_steps' iterations.
    """
    N, num_features = X.shape

    if initial_w is None:
        w = np.zeros(num_features)
    else:
        w = initial_w

    b = initial_b
    errors = []
    snapshots = []   # <-- NEW: store (b, w) over time

    for i in range(num_iterations):
        y_pred = b + np.dot(X, w)

        error = y_pred - y
        b_gradient = (2/N) * np.sum(error)
        w_gradient = (2/N) * np.dot(X.T, error)

        # Update
        b -= learning_rate * b_gradient
        w -= learning_rate * w_gradient

        # Record error
        mse = np.mean(error**2)
        errors.append(mse)

        # Record snapshot every N steps
        if i % record_steps == 0:
            snapshots.append((b, w.copy()))

    return b, w, errors, snapshots


# ========================
#          MAIN
# ========================






from matplotlib.animation import FuncAnimation

def animate_descent(points, snapshots):
    """
    Creates animation of the regression line improving over time.
    """
    x = points[:, 0]
    y = points[:, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="blue")

    line, = ax.plot([], [], color="red", linewidth=2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Gradient Descent Animation")

    x_vals = np.linspace(min(x), max(x), 200)

    def update(frame):
        b, w = snapshots[frame]
        y_vals = b + w[0] * x_vals
        line.set_data(x_vals, y_vals)
        ax.set_title(f"Iteration Snapshot {frame}")
        return line,

    ani = FuncAnimation(
        fig, update,
        frames=len(snapshots),
        interval=120,
        blit=True
    )

    plt.show()













def main():
    # ----- Hyperparameters -----
    data_path = "data.csv"
    learning_rate = 0.0001
    num_iterations = 1000

    # ----- Load data -----
    points = load_data(data_path)

    # ----- Train/Test split -----
    X = points[:, 0].reshape(-1, 1)
    X_scaled, X_mean, X_std = scale_features(X)
    y = points[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ----- Train our model -----
    b, w, errors, snapshots = gradient_descent(
        X_train,
        y_train,
        learning_rate=learning_rate,
        num_iterations=num_iterations
    )


    # Extract slope (1 feature setup)
    m = w[0]

    # ----- Predict on test data -----
    y_pred_test = predict(b, w, X_test)
    score = r2_score(y_test, y_pred_test)
    print(f"R^2 score on test data: {score:.4f}")

    print(f"Ending at b = {b:.4f}, m = {m:.4f}, training error = {errors[-1]:.6f}")

    # ----- PLots -----
    plot_regression(points, b, m)
    plot_error(errors)
    plot_residuals(y_test, y_pred_test)
    plot_correlation_heatmap(points)
    animate_descent(points, snapshots)

    # ----- SKLEARN COMPARISON -----
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)

    sk_b = model.intercept_
    sk_w = model.coef_
    y_pred_sklearn = model.predict(X_test)

    print("\n---- SKLEARN MODEL ----")
    print("Intercept:", sk_b)
    print("Weights:", sk_w)
    print("R^2:", r2_score(y_test, y_pred_sklearn))


if __name__ == "__main__":
    main()
