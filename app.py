from flask import (
    Flask,
    render_template_string,
    request,
    jsonify,
    send_from_directory,
    Response,
)
import os
import json
import datetime
import io
import base64
import matplotlib

# Use the non-interactive Agg backend for server-side plot generation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import csv
import sqlite3

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load data from CSV and SQLite for polynomial regression
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data.csv")
user_data_path = os.path.join(script_dir, "user_gpa_data.csv")
gpa_db_path = os.path.join(script_dir, "gpa_data.db")
reflect_db_path = os.path.join(script_dir, "reflection_data.db")
GRADE_TO_PCT = {"A": 92, "B": 78, "C": 62, "D": 47, "E": 25}
DIFFICULTY_TO_VALUE = {"Easy": 1, "Medium": 2, "Hard": 3}
DEFAULT_DIFFICULTY = "Medium"
average_grade = 0.0


def normalize_difficulty(value):
    #Normalize difficulty labels and return a valid difficulty string.
    if value is None or pd.isna(value):
        return DEFAULT_DIFFICULTY
    normalized = str(value).strip().title()
    return normalized if normalized in DIFFICULTY_TO_VALUE else DEFAULT_DIFFICULTY


def difficulty_value(value):
    #Convert a difficulty label into its numeric weight value.
    return DIFFICULTY_TO_VALUE.get(normalize_difficulty(value), DIFFICULTY_TO_VALUE[DEFAULT_DIFFICULTY])


def get_db_connection(db_path):
    #Open a SQLite connection with row access by column name.
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_gpa_database():
    #Initialize the GPA database and migrate any existing CSV records.
    conn = get_db_connection(gpa_db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpa_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            gpa REAL,
            num_subjects INTEGER,
            subjects TEXT NOT NULL
        )
        """
    )
    conn.commit()
    if cursor.execute("SELECT COUNT(*) FROM gpa_entries").fetchone()[0] == 0:
        migrate_user_csv_to_db(conn)
    conn.close()


def migrate_user_csv_to_db(conn):
    #Copy records from user_gpa_data.csv into the GPA SQLite database.
    if not os.path.isfile(user_data_path):
        return
    cursor = conn.cursor()
    try:
        with open(user_data_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                subjects = row.get("subjects")
                if not subjects:
                    continue
                timestamp = row.get("timestamp") or datetime.datetime.now().isoformat()
                gpa = float(row["gpa"]) if row.get("gpa") else None
                num_subjects = int(row["num_subjects"]) if row.get("num_subjects") else None
                cursor.execute(
                    "INSERT INTO gpa_entries (timestamp, gpa, num_subjects, subjects) VALUES (?, ?, ?, ?)",
                    (timestamp, gpa, num_subjects, subjects),
                )
        conn.commit()
    except Exception:
        # Ignore CSV migration issues and continue with an empty database.
        pass


def init_reflection_database():
    #Ensure reflection SQLite tables exist for predictions and training data.
    conn = get_db_connection(reflect_db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reflect_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            score REAL NOT NULL,
            max_mark REAL NOT NULL,
            predicted_pct REAL NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reflect_training (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            score REAL NOT NULL,
            max_mark REAL NOT NULL,
            pct_needed REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def load_training_data():
    #Load the training dataset from CSV and optionally from user GPA records.
    data = pd.read_csv(data_path)
    if "Grade" not in data.columns or "Unit" not in data.columns:
        raise ValueError("data.csv must contain 'Grade' and 'Unit' columns")
    if "Difficulty" not in data.columns:
        data["Difficulty"] = DEFAULT_DIFFICULTY
    else:
        data["Difficulty"] = (
            data["Difficulty"]
            .fillna(DEFAULT_DIFFICULTY)
            .astype(str)
            .str.strip()
            .str.title()
        )
        data.loc[~data["Difficulty"].isin(DIFFICULTY_TO_VALUE), "Difficulty"] = DEFAULT_DIFFICULTY

    data = data[["Grade", "Unit", "Difficulty"]].copy()
    data["DifficultyValue"] = data["Difficulty"].map(DIFFICULTY_TO_VALUE)

    extra_rows = []
    if os.path.isfile(gpa_db_path):
        try:
            conn = get_db_connection(gpa_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT subjects FROM gpa_entries")
            for row in cursor.fetchall():
                subjects = row["subjects"]
                if not subjects:
                    continue
                try:
                    entries = json.loads(subjects)
                except Exception:
                    continue
                for item in entries:
                    difficulty = normalize_difficulty(item.get("difficulty"))
                    units = item.get("units")
                    if units is None or pd.isna(units):
                        units = difficulty_value(difficulty)
                    pct = item.get("pct")
                    grade = item.get("grade")
                    if (
                        pct is not None
                        and isinstance(pct, (int, float))
                        and not np.isnan(pct)
                    ):
                        grade_value = pct
                    else:
                        grade_value = GRADE_TO_PCT.get(str(grade).upper())
                    if grade_value is None:
                        continue
                    extra_rows.append(
                        {
                            "Grade": grade_value,
                            "Unit": units,
                            "Difficulty": difficulty,
                        }
                    )
            conn.close()
        except Exception:
            # Ignore invalid user database rows and continue with base CSV data.
            pass
    elif os.path.isfile(user_data_path):
        try:
            user_df = pd.read_csv(user_data_path)
            for _, row in user_df.iterrows():
                subjects = row.get("subjects")
                if not subjects or pd.isna(subjects):
                    continue
                try:
                    entries = json.loads(subjects)
                except Exception:
                    continue
                for item in entries:
                    difficulty = normalize_difficulty(item.get("difficulty"))
                    units = item.get("units")
                    if units is None or pd.isna(units):
                        units = difficulty_value(difficulty)
                    pct = item.get("pct")
                    grade = item.get("grade")
                    if (
                        pct is not None
                        and isinstance(pct, (int, float))
                        and not np.isnan(pct)
                    ):
                        grade_value = pct
                    else:
                        grade_value = GRADE_TO_PCT.get(str(grade).upper())
                    if grade_value is None:
                        continue
                    extra_rows.append(
                        {
                            "Grade": grade_value,
                            "Unit": units,
                            "Difficulty": difficulty,
                        }
                    )
        except Exception:
            pass

    if extra_rows:
        user_data = pd.DataFrame(extra_rows)
        data = pd.concat([data, user_data], ignore_index=True)

    data["Difficulty"] = (
        data["Difficulty"].fillna(DEFAULT_DIFFICULTY).astype(str).str.strip().str.title()
    )
    data.loc[~data["Difficulty"].isin(DIFFICULTY_TO_VALUE), "Difficulty"] = DEFAULT_DIFFICULTY
    data["DifficultyValue"] = data["Difficulty"].map(DIFFICULTY_TO_VALUE)
    data["GradeAboveAverage"] = data["Grade"] - data["Grade"].mean()

    return data


def retrain_model():
    #Build and fit the polynomial regression model from the current training data.
    global data, X, y, poly, model, average_grade
    data = load_training_data()
    average_grade = data["Grade"].mean()
    X = data[["Grade", "DifficultyValue", "GradeAboveAverage"]].values
    y = data["Unit"].values
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)


init_gpa_database()
init_reflection_database()
retrain_model()


def fig_to_png_bytes(fig):
    #Convert a Matplotlib figure to PNG bytes for HTTP responses.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


@app.route("/admin")
def admin():
    # Serve the admin HTML page from the project root.
    return send_from_directory(".", "admin.html")


@app.route("/admin/plot/<chart_name>")
def admin_plot(chart_name):
    #Render admin chart images for selected training diagnostics.
    if chart_name == "score-vs-gpa":
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X[:, 0], y, color="#2d5be3", edgecolor="k", s=75, alpha=0.9)
        ax.set_title("Training data: Grade vs Unit")
        ax.set_xlabel("Grade")
        ax.set_ylabel("Unit")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        ax.set_ylim(min(y.min() - 0.5, 0), y.max() + 0.5)
    elif chart_name == "polynomial-fit":
        x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 200).reshape(-1, 1)
        difficulty_line = np.full((x_line.shape[0], 1), DIFFICULTY_TO_VALUE[DEFAULT_DIFFICULTY])
        grade_above_avg_line = x_line - average_grade
        x_line_features = np.hstack([x_line, difficulty_line, grade_above_avg_line])
        y_line = model.predict(poly.transform(x_line_features))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            X[:, 0],
            y,
            color="#2d5be3",
            edgecolor="k",
            s=75,
            alpha=0.9,
            label="Training data",
        )
        ax.plot(
            x_line.flatten(),
            y_line,
            color="#f59e0b",
            linewidth=3,
            label="Polynomial fit",
        )
        ax.set_title("Polynomial regression fit")
        ax.set_xlabel("Grade")
        ax.set_ylabel("Unit")
        ax.legend()
        ax.grid(True, alpha=0.25)
        ax.set_xlim(X.min() - 1, X.max() + 1)
        ax.set_ylim(min(y.min() - 0.5, 0), y.max() + 0.5)
    else:
        return "Chart not found", 404

    png = fig_to_png_bytes(fig)
    return Response(png, mimetype="image/png")


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
    #Predict unit requirements for a given grade and difficulty.
    grade = request.args.get("grade", type=float)
    if grade is None:
        return "Please provide a grade parameter, e.g. /predict?grade=85"
    difficulty = request.args.get("difficulty", default=DEFAULT_DIFFICULTY, type=str)
    difficulty_val = difficulty_value(difficulty)
    grade_above_average = grade - average_grade
    prediction_features = [[grade, difficulty_val, grade_above_average]]
    predicted_unit = model.predict(poly.transform(prediction_features))[0]
    return (
        f"Predicted Unit for Grade {grade} "
        f"(difficulty={normalize_difficulty(difficulty)}, "
        f"aboveAvg={grade_above_average:.2f}): {predicted_unit:.2f}"
    )


@app.route("/log_gpa", methods=["POST"])
def log_gpa():
    #Persist GPA log entries to the database and retrain the model.
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    gpa = data.get("gpa")
    subjects = data.get("subjects", [])
    timestamp = datetime.datetime.now().isoformat()

    # Prepare CSV row
    subjects_json = json.dumps(subjects)
    conn = get_db_connection(gpa_db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO gpa_entries (timestamp, gpa, num_subjects, subjects) VALUES (?, ?, ?, ?)",
        (timestamp, gpa, len(subjects), subjects_json),
    )
    conn.commit()
    conn.close()

    retrain_model()
    return jsonify({"status": "logged"}), 200


@app.route("/reflect_predictions.csv")
def reflect_predictions_csv():
    #Serve reflection prediction CSV output from the database
    if os.path.isfile(reflect_db_path):
        conn = get_db_connection(reflect_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT score, max_mark, predicted_pct FROM reflect_predictions")
        rows = cursor.fetchall()
        conn.close()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["score", "max_mark", "predicted_pct"])
        writer.writerows(rows)
        return Response(output.getvalue(), mimetype="text/csv")

    if os.path.isfile(os.path.join(script_dir, "reflect_predictions.csv")):
        return send_from_directory(script_dir, "reflect_predictions.csv")

    return Response("score,max_mark,predicted_pct\n", mimetype="text/csv")


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Scorely backend running → http://localhost:5000")
    app.run(debug=True, port=5000)
