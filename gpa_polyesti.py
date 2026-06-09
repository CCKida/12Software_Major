"""
Average Calculator (Weighted) for NSW Years 7-12

This script computes the weighted average of grades using difficulty weights
from `data.csv` instead of unit weights. Difficulty values are mapped to
numeric weights where Easy=1, Medium=2, Hard=3, and the weighted average
is calculated from those difficulty-based weights.

Expect `data.csv` to contain at least the columns `Grade` and `Difficulty`.
If `Difficulty` is missing the default is Medium.
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "data.csv")
gpa_db_path = os.path.join(script_dir, "gpa_data.db")
user_csv_path = os.path.join(script_dir, "user_gpa_data.csv")


GRADE_TO_PCT = {"A": 92, "B": 78, "C": 62, "D": 47, "E": 25}
DIFFICULTY_TO_VALUE = {"Easy": 1, "Medium": 2, "Hard": 3}
DEFAULT_DIFFICULTY = "Medium"


def normalize_difficulty(value):
    #Return a valid difficulty string for a normalized difficulty label
    if value is None or pd.isna(value):
        return DEFAULT_DIFFICULTY
    normalized = str(value).strip().title()
    return normalized if normalized in DIFFICULTY_TO_VALUE else DEFAULT_DIFFICULTY


def load_user_rows_from_db(db_path):
    #Load additional user rows from the GPA SQLite database.
    extra_rows = []
    if not os.path.isfile(db_path):
        return extra_rows
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT subjects FROM gpa_entries")
        for (subjects,) in cursor.fetchall():
            if not subjects:
                continue
            try:
                entries = json.loads(subjects)
            except Exception:
                continue
            for item in entries:
                pct = item.get("pct")
                grade = item.get("grade")
                if pct is not None and isinstance(pct, (int, float)) and not np.isnan(pct):
                    grade_value = pct
                else:
                    grade_value = GRADE_TO_PCT.get(str(grade).upper())
                if grade_value is None:
                    continue
                difficulty = normalize_difficulty(item.get("difficulty"))
                extra_rows.append(
                    {
                        "Grade": grade_value,
                        "Difficulty": difficulty,
                        "Unit": item.get("units") if item.get("units") is not None else np.nan,
                    }
                )
        conn.close()
    except Exception:
        # Ignore malformed database entries and continue.
        pass
    return extra_rows


def load_user_rows_from_csv(csv_path):
    #Load additional user rows from the CSV file if the database is not present.
    extra_rows = []
    if not os.path.isfile(csv_path):
        return extra_rows
    try:
        user_df = pd.read_csv(csv_path)
        for _, row in user_df.iterrows():
            subjects = row.get("subjects")
            if not subjects or pd.isna(subjects):
                continue
            try:
                entries = json.loads(subjects)
            except Exception:
                continue
            for item in entries:
                pct = item.get("pct")
                grade = item.get("grade")
                if pct is not None and isinstance(pct, (int, float)) and not np.isnan(pct):
                    grade_value = pct
                else:
                    grade_value = GRADE_TO_PCT.get(str(grade).upper())
                if grade_value is None:
                    continue
                difficulty = normalize_difficulty(item.get("difficulty"))
                extra_rows.append(
                    {
                        "Grade": grade_value,
                        "Difficulty": difficulty,
                        "Unit": item.get("units") if item.get("units") is not None else np.nan,
                    }
                )
    except Exception:
        pass
    return extra_rows


def load_data(path):
    #Load base grade data and enrich it with user-provided records.
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "Grade" not in df.columns:
        raise ValueError("data.csv must contain a 'Grade' column")
    if "Difficulty" not in df.columns:
        df["Difficulty"] = DEFAULT_DIFFICULTY
    else:
        df["Difficulty"] = (
            df["Difficulty"].fillna(DEFAULT_DIFFICULTY).astype(str).str.strip().str.title()
        )
        df.loc[~df["Difficulty"].isin(DIFFICULTY_TO_VALUE), "Difficulty"] = DEFAULT_DIFFICULTY

    df["DifficultyVal"] = df["Difficulty"].map(DIFFICULTY_TO_VALUE)

    extra_rows = load_user_rows_from_db(gpa_db_path)
    if not extra_rows:
        extra_rows = load_user_rows_from_csv(user_csv_path)

    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
        extra_df["Difficulty"] = extra_df["Difficulty"].fillna(DEFAULT_DIFFICULTY).astype(str).str.strip().str.title()
        extra_df.loc[~extra_df["Difficulty"].isin(DIFFICULTY_TO_VALUE), "Difficulty"] = DEFAULT_DIFFICULTY
        extra_df["DifficultyVal"] = extra_df["Difficulty"].map(DIFFICULTY_TO_VALUE)
        df = pd.concat([df, extra_df], ignore_index=True)

    return df.copy()


def weighted_average(grades, weights):
    #Compute a weighted average using numeric weights.
    grades = np.asarray(grades, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total_weight = np.sum(weights)
    if total_weight == 0:
        return float("nan"), 0.0, 0.0
    total_weighted = np.sum(grades * weights)
    return total_weighted / total_weight, total_weighted, total_weight


def main():
    #Main entry point: compute weighted average, train estimator, and generate a plot.
    df = load_data(csv_path)
    
    # Use difficulty values as weights instead of unit values.
    avg, total_weighted, total_weight = weighted_average(df["Grade"].values, df["DifficultyVal"].values)

    print("=" * 60)
    print("SIMPLE WEIGHTED AVERAGE CALCULATOR")
    print("=" * 60)
    print(f"Total Weighted Score: {total_weighted}")
    print(f"Total Difficulty Weight: {total_weight}")
    if np.isnan(avg):
        print("Weighted average: undefined (total difficulty is zero)")
    else:
        print(f"Weighted average: {avg:.2f}")
    print("=" * 60)

    # --- Polynomial estimator (scikit-learn) ---
    # Predict difficulty from score, optionally using unit data as an additional feature.
    use_units = 'Unit' in df.columns
    if use_units:
        X = np.column_stack([df['Grade'].values, df['Unit'].values])
    else:
        X = df['Grade'].values.reshape(-1, 1)
    y = df['DifficultyVal'].values

    # Split data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Polynomial features degree 2
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    preds = model.predict(X_test_poly)
    r2 = r2_score(y_test, preds)
    print(f"Polynomial estimator (degree=2) Test R²: {r2:.4f}")

    # Plot: scatter and polynomial fit line (for Grade dimension)
    grade_min, grade_max = df['Grade'].min(), df['Grade'].max()
    grade_line = np.linspace(grade_min, grade_max, 300).reshape(-1, 1)
    if use_units:
        # use median unit value for plotting as second feature
        median_unit = np.nanmedian(df['Unit'].values)
        unit_col = np.full((grade_line.shape[0], 1), median_unit)
        X_line = np.hstack([grade_line, unit_col])
    else:
        X_line = grade_line

    y_line = model.predict(poly.transform(X_line))

    plt.figure(figsize=(10, 6))
    plt.scatter(df['Grade'], df['DifficultyVal'], label='Data (difficulty)', color='blue', alpha=0.7)
    plt.plot(grade_line, y_line, color='red', linewidth=2, label='Polynomial fit (deg 2)')
    plt.title('Polynomial Regression: Score → Difficulty')
    plt.xlabel('Score (Grade)')
    plt.ylabel('Difficulty (1=Easy,2=Medium,3=Hard)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Annotate stats
    text = f"Weighted avg: {avg:.2f}\nTest R²: {r2:.4f}"
    plt.gca().text(0.02, 0.98, text, transform=plt.gca().transAxes, va='top')

    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpa_polyesti_plot.png')
    plt.savefig(out_png)
    print(f"Plot saved to: {out_png}")
    plt.show()


if __name__ == '__main__':
    main()
