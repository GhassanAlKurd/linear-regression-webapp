import numpy as np
import csv

def generate_clean_data(num_points=100, slope=3.5, intercept=10, noise_std=2):
    """
    Generate a clean synthetic dataset following:
    y = slope * x + intercept + noise
    """
    x = np.linspace(0, 100, num_points)
    noise = np.random.normal(0, noise_std, num_points)
    y = slope * x + intercept + noise
    data = np.column_stack((x, y))

    with open("data_clean.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print("Generated data_clean.csv successfully!")

if __name__ == "__main__":
    generate_clean_data()
