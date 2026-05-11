from flask import Flask, render_template_string, request, jsonify, send_from_directory, Response
import os
import json
import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import csv

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load data from CSV for polynomial regression
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data.csv')
user_data_path = os.path.join(script_dir, 'user_gpa_data.csv')
GRADE_TO_PCT = {'A': 92, 'B': 78, 'C': 62, 'D': 47, 'E': 25}


def load_training_data():
    data = pd.read_csv(data_path)
    if 'Grade' not in data.columns or 'Unit' not in data.columns:
        raise ValueError("data.csv must contain 'Grade' and 'Unit' columns")
    data = data[['Grade', 'Unit']].copy()

    if os.path.isfile(user_data_path):
        try:
            user_df = pd.read_csv(user_data_path)
            extra_rows = []
            for _, row in user_df.iterrows():
                subjects = row.get('subjects')
                if not subjects or pd.isna(subjects):
                    continue
                try:
                    entries = json.loads(subjects)
                except Exception:
                    continue
                for item in entries:
                    units = item.get('units')
                    if units is None or pd.isna(units):
                        continue
                    pct = item.get('pct')
                    grade = item.get('grade')
                    if pct is not None and isinstance(pct, (int, float)) and not np.isnan(pct):
                        grade_value = pct
                    else:
                        grade_value = GRADE_TO_PCT.get(str(grade).upper())
                    if grade_value is None:
                        continue
                    extra_rows.append({'Grade': grade_value, 'Unit': units})
            if extra_rows:
                user_data = pd.DataFrame(extra_rows)
                data = pd.concat([data, user_data], ignore_index=True)
        except Exception:
            pass

    return data


def retrain_model():
    global data, X, y, poly, model
    data = load_training_data()
    X = data[['Grade']].values
    y = data['Unit'].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)


retrain_model()


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

@app.route("/admin")
def admin():
    return send_from_directory(".", "admin.html")

@app.route("/admin/plot/<chart_name>")
def admin_plot(chart_name):
    if chart_name == 'score-vs-gpa':
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X.flatten(), y, color='#2d5be3', edgecolor='k', s=75, alpha=0.9)
        ax.set_title('Training data: Grade vs Unit')
        ax.set_xlabel('Grade')
        ax.set_ylabel('Unit')
        ax.grid(True, alpha=0.25)
        ax.set_xlim(X.min() - 1, X.max() + 1)
        ax.set_ylim(min(y.min() - 0.5, 0), y.max() + 0.5)
    elif chart_name == 'polynomial-fit':
        x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_line = model.predict(poly.transform(x_line))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X.flatten(), y, color='#2d5be3', edgecolor='k', s=75, alpha=0.9, label='Training data')
        ax.plot(x_line.flatten(), y_line, color='#f59e0b', linewidth=3, label='Polynomial fit')
        ax.set_title('Polynomial regression fit')
        ax.set_xlabel('Grade')
        ax.set_ylabel('Unit')
        ax.legend()
        ax.grid(True, alpha=0.25)
        ax.set_xlim(X.min() - 1, X.max() + 1)
        ax.set_ylim(min(y.min() - 0.5, 0), y.max() + 0.5)
    else:
        return "Chart not found", 404

    png = fig_to_png_bytes(fig)
    return Response(png, mimetype='image/png')

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve the static homepage."""
    return send_from_directory(".", "index.html")

@app.route("/gpa")
def gpa():
    """Serve the static GPA calculator page."""
    return send_from_directory(".", "gpa.html")

@app.route("/reflect")
def reflect():
    """Serve the static reflection form page."""
    return send_from_directory(".", "reflect.html")

@app.route("/predict")
def predict():
    grade = request.args.get('grade', type=float)
    if grade is None:
        return "Please provide a grade parameter, e.g. /predict?grade=85"
    grade_poly = poly.transform([[grade]])
    predicted_unit = model.predict(grade_poly)[0]
    return f"Predicted Unit for Grade {grade}: {predicted_unit:.2f}"

@app.route("/log_gpa", methods=["POST"])
def log_gpa():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    gpa = data.get('gpa')
    subjects = data.get('subjects', [])
    timestamp = datetime.datetime.now().isoformat()

    # Prepare CSV row
    row = {
        'timestamp': timestamp,
        'gpa': gpa,
        'num_subjects': len(subjects),
        'subjects': json.dumps(subjects)  # Store as JSON string
    }

    file_exists = os.path.isfile(user_data_path)

    with open(user_data_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'gpa', 'num_subjects', 'subjects']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    retrain_model()
    return jsonify({"status": "logged"}), 200

# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Scorely backend running → http://localhost:5000")
    app.run(debug=True, port=5000)
