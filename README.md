🔥 Linear Regression from Scratch (Interactive Web App)

An end-to-end implementation of linear regression built entirely from scratch, without using scikit-learn for training. The project includes:

A full gradient descent optimizer

Feature scaling

Residual analysis

An interactive Streamlit web app for visualization

A fully deployed app on Streamlit Cloud

This project is ideal for learning how linear regression actually works under the hood — beyond the black box of machine-learning libraries.

📊 Features
🛠 Core Algorithm

Implements gradient descent manually (no sklearn training)

Trains on data from data.csv (two columns: x, y)

Includes cost function, parameter updates, and convergence tracking

📈 Visualizations

The app provides multiple interactive visual outputs:

Scatter plot of original data

Regression line (based on learned slope/intercept)

Error curve (MSE vs. iterations)

Residual plot to evaluate model fit

Correlation heatmap

User prediction tool (enter X → get predicted Y)

🌐 Web App (Streamlit)

Fully interactive sliders for learning rate & iterations

Real-time model retraining

Visual feedback for every update

Cloud-hosted version available via shareable URL

🚀 Live Demo

👉 Streamlit App:
https://linear-regression-webapp-gxx6dmqhmdhcswgjcwa9x.streamlit.app/

📂 Project Structure
Linear_Regression_Project/
│── app.py                 # Streamlit web app
│── linear_regression.py   # Gradient descent + math logic
│── data.csv               # Raw dataset
│── data_clean.csv         # Cleaned dataset
│── generate_clean_data.py # Data cleaning script
│── requirements.txt       # Dependencies
│── README.md              # Project documentation

🧠 How the Algorithm Works

Initialize parameters

Intercept b = 0

Weight(s) w = 0

Scale features (Z-score normalization)

Run gradient descent

Compute predictions

Compute gradients

Update parameter values

Track error per iteration

Evaluate model

R² score

Residuals

Visualizations

Compare with scikit-learn

After training from scratch, the app also trains a real sklearn model

Outputs both sets of results for comparison

🖥️ Run Locally
1. Create & activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# OR
.\.venv\Scripts\activate       # Windows

2. Install dependencies
pip install -r requirements.txt

3. Run Streamlit app
streamlit run app.py

🧩 Tech Used

Python 3.11

NumPy — vectorized math

Matplotlib / Seaborn — visualizations

Streamlit — interactive UI

scikit-learn — baseline comparison model