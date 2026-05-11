# Year 12 Software Automation: Polynomial Regression Estimator
# This script loads the CSV dataset and fits a polynomial model to predict unit values from score values.
# It also calculates the weighted average GPA based on units and scores.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load historical data from CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'data.csv')
df = pd.read_csv(csv_path)
# Trim whitespace from column names in case headers have extra spaces
df.columns = df.columns.str.strip()

# Check for correct column names
if 'Grade' not in df.columns or 'Unit' not in df.columns:
    raise ValueError("data.csv must contain 'Grade' and 'Unit' columns")

scores = df['Grade'].values
units = df['Unit'].values

# Step 2: Calculate the weighted average GPA
def calculate_weighted_average(scores, units):
    total_weighted_score = np.sum(scores * units)
    total_units = np.sum(units)
    if total_units == 0:
        return 0
    return total_weighted_score / total_units

average_gpa = calculate_weighted_average(scores, units)

# Step 3: Fit a polynomial model using NumPy
# degree = 2 means y = a*x^2 + b*x + c
degree = 2
coeffs = np.polyfit(scores, units, degree)

# Helper to predict unit from score
def predict_unit(score_value):
    return np.polyval(coeffs, score_value)

# Step 4: Print the fitted polynomial equation
print('=' * 70)
print('POLYNOMIAL REGRESSION MODEL (Degree 2)')
print('=' * 70)
print(f'Coefficients: {coeffs}')
print('Equation:')
print(f'Unit = {coeffs[0]:.6f}×x² + {coeffs[1]:.6f}×x + {coeffs[2]:.6f}')
print('where x = score')
print('=' * 70)
print()

# Step 5: Print the calculated average GPA
print('=' * 70)
print('WEIGHTED AVERAGE GPA CALCULATION')
print('=' * 70)
print(f'Total Weighted Score: {np.sum(scores * units)}')
print(f'Total Units: {np.sum(units)}')
print(f'Average GPA: {average_gpa:.2f}')
print('=' * 70)
print()

# Step 6: Predict a sample score
sample_score = 92
predicted_unit = predict_unit(sample_score)
print('=' * 70)
print('PREDICTION FOR SCORE')
print('=' * 70)
print(f'Score: {sample_score}')
print(f'Predicted Unit: {predicted_unit:.4f}')
print('=' * 70)
print()

# Step 5: Visualize the data and polynomial fit
x_line = np.linspace(scores.min(), scores.max(), 200)
y_line = predict_unit(x_line)

plt.figure(figsize=(10, 6))
plt.scatter(scores, units, color='blue', label='Training data', s=80, alpha=0.8)
plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Polynomial fit (degree {degree})')
plt.scatter(sample_score, predicted_unit, color='green', s=150, marker='*', label='Sample prediction', zorder=5)

plt.title('Polynomial Regression: Score → Unit')
plt.xlabel('Score')
plt.ylabel('Unit')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
