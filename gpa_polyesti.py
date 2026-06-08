"""
Average Calculator (Weighted) for NSW Years 7-12

This script computes the weighted average of scores using the corresponding
weights (units/credits) from `data.csv`. It's a simplified replacement of
the previous polynomial/GPA estimator and prints the total weighted score,
total weight and the weighted average (percentage).

Expect `data.csv` to contain at least the columns `Grade` and `Unit`.
"""

import os
import numpy as np
import pandas as pd


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "data.csv")


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "Grade" not in df.columns or "Unit" not in df.columns:
        raise ValueError("data.csv must contain 'Grade' and 'Unit' columns")
    return df[["Grade", "Unit"]].copy()


def weighted_average(grades, weights):
    grades = np.asarray(grades, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total_weight = np.sum(weights)
    if total_weight == 0:
        return float("nan"), 0.0, 0.0
    total_weighted = np.sum(grades * weights)
    return total_weighted / total_weight, total_weighted, total_weight


def main():
    df = load_data(csv_path)
    avg, total_weighted, total_weight = weighted_average(df["Grade"].values, df["Unit"].values)

    print("=" * 60)
    print("SIMPLE WEIGHTED AVERAGE CALCULATOR")
    print("=" * 60)
    print(f"Total Weighted Score: {total_weighted}")
    print(f"Total Weight (units): {total_weight}")
    if np.isnan(avg):
        print("Weighted average: undefined (total weight is zero)")
    else:
        print(f"Weighted average: {avg:.2f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
