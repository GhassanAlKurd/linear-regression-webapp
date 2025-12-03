# 🔥 Linear Regression from Scratch (Interactive Web App)

An end-to-end implementation of **linear regression built entirely from scratch**, without using scikit-learn for training.  
Includes:

- A full gradient descent optimizer  
- Feature scaling  
- Residual analysis  
- An interactive **Streamlit web app** for visualization  
- A fully deployed version on Streamlit Cloud  

This project is perfect for learning how linear regression actually works under the hood — beyond the black-box of machine learning libraries.

---

## 🚀 Features

### 🧠 Core Algorithm
- Implements gradient descent manually (no scikit-learn training)
- Train on data directly from `data.csv`
- Includes:
  - Parameter updates  
  - Error curves  
  - Convergence tracking  

### 📊 Visualizations
- Scatter plot of original data  
- Best-fit regression line  
- Regression line (based on learned slope/intercept)  
- Error curve (MSE vs. iterations)  
- Residuals plot  
- Correlation heatmap  

### 🌐 Web App (Frontend)
- Built with **Streamlit**
- Clean sidebar controls for:
  - Learning rate  
  - Training iterations  
- Realtime retraining on every update  
- Cloud-hosted version available via shareable URL  

---

## 🌍 Live Demo

👉 **Web App:**  
https://linear-regression-webapp-gxx6dqhmhdcnhdsvwgjcwva9x.streamlit.app/

---

## 🗂️ Project Structure

Linear_Regression_Project/
│── app.py                 # Streamlit web app  
│── linear_regression.py   # Gradient descent + math logic  
│── data.csv               # Raw dataset  
│── data_clean.csv         # Cleaned dataset (optional)  
│── generate_clean_data.py # Data cleaning script  
│── requirements.txt       # Dependencies  
│── README.md              # Project documentation

---

## 📘 How the Algorithm Works

### 🧮 Initialization
- `b = 0` (intercept)  
- `w = 0` (weight/slope)  

### 🔄 Training Loop
1. Scale features (Z-score normalization)  
2. Run gradient descent  
3. Compute predictions  
4. Update parameters  
5. Track error per iteration  

### 📈 Metrics
- Final model parameters  
- R² score  
- Residuals  
- MSE trend  
- Comparison with scikit-learn model  

---



