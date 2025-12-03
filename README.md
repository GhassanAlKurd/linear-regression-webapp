# Linear Regression from Scratch

This project implements simple linear regression using **gradient descent** in
Python, with **NumPy** for fast numerical operations and **Matplotlib** for
visualization.

## Features

- Loads data from `data.csv` (two columns: x, y)
- Implements gradient descent from scratch (no scikit-learn)
- Plots:
  - Original data + best-fit regression line
  - Error (mean squared error) vs. iterations

## How to Run

> **Python version:** Streamlit (and its `pyarrow` dependency) currently ships
> prebuilt wheels only for Python 3.7–3.12. Use Python 3.11 for the smoothest
> experience on this project.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py   # or: python linear_regression.py
```
