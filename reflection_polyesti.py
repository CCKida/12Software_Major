"""
Generate a training CSV that contains: score, max_mark, pct_needed
Train a polynomial regression (degree 2) to predict pct_needed from score and max_mark
Save the dataset and predictions to CSV files.
"""
import os
import json
import numpy as np
import pandas as pd


GRADE_TO_PCT = {'A': 92, 'B': 78, 'C': 62, 'D': 47, 'E': 25}


def load_scores_from_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if 'Grade' in df.columns:
        scores = df['Grade'].values
    elif 'Score' in df.columns:
        scores = df['Score'].values
    else:
        raise ValueError("No 'Grade' or 'Score' column found in data.csv")
    return scores


def load_from_user_gpa(user_csv_path):
    rows = []
    if not os.path.isfile(user_csv_path):
        return rows
    try:
        user_df = pd.read_csv(user_csv_path)
    except Exception:
        return rows
    for _, row in user_df.iterrows():
        subjects = row.get('subjects')
        if pd.isna(subjects) or not subjects:
            continue
        try:
            entries = json.loads(subjects)
        except Exception:
            continue
        for item in entries:
            units = item.get('units')
            pct = item.get('pct')
            grade = item.get('grade')
            if pct is not None and not pd.isna(pct):
                score = float(pct)
            else:
                score = GRADE_TO_PCT.get(str(grade).upper())
            if score is None:
                continue
            rows.append(score)
    return rows


def build_dataset(base_dir):
    data_csv = os.path.join(base_dir, 'data.csv')
    user_csv = os.path.join(base_dir, 'user_gpa_data.csv')
    scores = []
    if os.path.isfile(data_csv):
        try:
            scores = load_scores_from_data(data_csv)
        except Exception:
            scores = []
    user_scores = load_from_user_gpa(user_csv)
    if len(user_scores) > 0:
        scores = np.concatenate([np.asarray(scores, dtype=float), np.asarray(user_scores, dtype=float)])
    scores = np.asarray(scores, dtype=float)
    # If no scores found, create a small synthetic dataset
    if scores.size == 0:
        scores = np.array([55, 65, 72, 80, 88, 92, 45, 30, 100])

    # Default max marks: assume 100 for all entries
    max_marks = np.full_like(scores, 100.0)

    pct_needed = np.clip((max_marks - scores) / max_marks * 100.0, 0.0, 100.0)

    df = pd.DataFrame({'score': scores, 'max_mark': max_marks, 'pct_needed': pct_needed})
    return df


def poly_features_two_vars(x1, x2, degree=2):
    # Create polynomial features for two variables up to given degree
    # Order: 1, x1, x2, x1^2, x1*x2, x2^2 (for degree=2)
    if degree != 2:
        raise NotImplementedError('Only degree=2 implemented')
    x1 = np.asarray(x1).reshape(-1)
    x2 = np.asarray(x2).reshape(-1)
    X = np.vstack([
        np.ones_like(x1),
        x1,
        x2,
        x1 ** 2,
        x1 * x2,
        x2 ** 2,
    ]).T
    return X


def train_poly_regression(df):
    X = poly_features_two_vars(df['score'].values, df['max_mark'].values, degree=2)
    y = df['pct_needed'].values
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs


def predict_from_coeffs(df, coeffs):
    X = poly_features_two_vars(df['score'].values, df['max_mark'].values, degree=2)
    preds = X.dot(coeffs)
    preds = np.clip(preds, 0.0, 100.0)
    return preds


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = build_dataset(base_dir)
    out_path = os.path.join(base_dir, 'reflect_training.csv')
    df.to_csv(out_path, index=False)
    print(f'Wrote training dataset to: {out_path}')

    coeffs = train_poly_regression(df)
    print('Trained polynomial regression coefficients:')
    print(coeffs)

    df['predicted_pct'] = predict_from_coeffs(df, coeffs)
    pred_out = os.path.join(base_dir, 'reflect_predictions.csv')
    df.to_csv(pred_out, index=False)
    print(f'Wrote predictions to: {pred_out}')


if __name__ == '__main__':
    main()
