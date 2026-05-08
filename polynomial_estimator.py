# Year 12 Software Automation: Polynomial Regression Mark Estimator
# This script uses Polynomial Regression to predict a student's missing final exam mark
# based on their mid-term mark. Unlike Linear Regression, this can model curved relationships.

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Step 1: Load historical data from CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'data.csv')
df = pd.read_csv(csv_path)
subject_one = df['mid_term_mark'].values
subject_two = df['final_exam_mark'].values
subject names = df['subject_name'].values

# Step 2: Prepare the data for scikit-learn
X = subject_one.reshape(-1, 1)  # Input features (mid-term marks)
y = subject_two  # Target values (final exam marks)

# Step 3: Create polynomial features
# degree=2 means we'll use: 1, x, x^2
# This allows the model to fit a curved line (parabola) instead of just a straight line
degree = 2
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# What does X_poly look like?
# If X = [[65], [70], [75], ...]
# Then X_poly = [[1, 65, 4225], [1, 70, 4900], [1, 75, 5625], ...]
#                 ^   ^    ^
#                 1   x   x^2

# Step 4: Create and train the Polynomial Regression model
# Note: Polynomial regression still uses LinearRegression!
# It's "linear" in the coefficients, not in the features
model = LinearRegression()
model.fit(X_poly, y)

# Step 5: Display the model equation
print("=" * 70)
print("POLYNOMIAL REGRESSION MODEL (Degree 2)")
print("=" * 70)
print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficients: {model.coef_}")
print()
print("Equation:")
print(f"Final Mark = {model.intercept_:.4f} + ({model.coef_[1]:.4f} × x) + ({model.coef_[2]:.4f} × x²)")
print("where x = mid-term mark")
print("=" * 70)
print()

# Step 6: Predict Alex's missing final exam mark
# Using 92 to demonstrate the plateau effect at high marks
alex_midterm = 92
alex_midterm_reshaped = np.array([[alex_midterm]])
alex_midterm_poly = poly_features.transform(alex_midterm_reshaped)
alex_predicted_final = model.predict(alex_midterm_poly)[0]

# Step 7: Display the prediction
print("=" * 70)
print("MARK PREDICTION FOR ALEX")
print("=" * 70)
print(f"Mid-term Mark: {alex_midterm}")
print(f"Predicted Final Exam Mark: {alex_predicted_final:.2f}")
print("=" * 70)
print()

# Step 8: Compare Linear vs Polynomial Regression
# Train a simple linear model for comparison
linear_model = LinearRegression()
linear_model.fit(X, y)
alex_linear_prediction = linear_model.predict(alex_midterm_reshaped)[0]

print("=" * 70)
print("COMPARISON: LINEAR vs POLYNOMIAL REGRESSION")
print("=" * 70)
print(f"Linear Regression Prediction:     {alex_linear_prediction:.2f}")
print(f"Polynomial Regression Prediction: {alex_predicted_final:.2f}")
print(f"Difference:                        {abs(alex_predicted_final - alex_linear_prediction):.2f}")
print("=" * 70)

# Step 9: Visualize the data and both models
plt.figure(figsize=(12, 5))

# Left plot: Polynomial Regression
plt.subplot(1, 2, 1)
plt.scatter(mid_term_marks, final_exam_marks, color='blue', label='Historical Data', s=100)

# Create smooth curve for polynomial regression
x_line = np.linspace(mid_term_marks.min(), mid_term_marks.max(), 100).reshape(-1, 1)
x_line_poly = poly_features.transform(x_line)
y_poly_line = model.predict(x_line_poly)
plt.plot(x_line, y_poly_line, color='red', linewidth=2, label=f'Polynomial (degree {degree})')

plt.scatter(alex_midterm, alex_predicted_final, color='green', s=300, 
            marker='*', label="Alex's Prediction", zorder=5)

plt.xlabel('Mid-term Mark', fontsize=12)
plt.ylabel('Final Exam Mark', fontsize=12)
plt.title('Polynomial Regression', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Right plot: Comparison of Linear vs Polynomial
plt.subplot(1, 2, 2)
plt.scatter(mid_term_marks, final_exam_marks, color='blue', label='Historical Data', s=100)

# Linear regression line
y_linear_line = linear_model.predict(x_line)
plt.plot(x_line, y_linear_line, color='orange', linewidth=2, label='Linear', linestyle='--')

# Polynomial regression curve
plt.plot(x_line, y_poly_line, color='red', linewidth=2, label=f'Polynomial (degree {degree})')

plt.scatter(alex_midterm, alex_predicted_final, color='green', s=300, 
            marker='*', label="Alex's Prediction (Poly)", zorder=5)
plt.scatter(alex_midterm, alex_linear_prediction, color='purple', s=200, 
            marker='D', label="Alex's Prediction (Linear)", zorder=5)

plt.xlabel('Mid-term Mark', fontsize=12)
plt.ylabel('Final Exam Mark', fontsize=12)
plt.title('Linear vs Polynomial Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
