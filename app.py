from flask import Flask, render_template_string, request, jsonify, send_from_directory, Response
import os
import json
import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load data from CSV for polynomial regression
data = pd.read_csv('data.csv')
X = data[['score']].values
y = data['unit'].values

# Fit a polynomial model (degree 2) with NumPy
coeffs = np.polyfit(X.flatten(), y, 2)

def predict_gpa(score):
    return np.polyval(coeffs, score)


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
        ax.set_title('Training data: Score vs GPA')
        ax.set_xlabel('Score (%)')
        ax.set_ylabel('GPA')
        ax.grid(True, alpha=0.25)
        ax.set_xlim(40, 105)
        ax.set_ylim(0.8, 4.2)
    elif chart_name == 'polynomial-fit':
        x_line = np.linspace(X.min(), X.max(), 200)
        y_line = np.polyval(coeffs, x_line)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X.flatten(), y, color='#2d5be3', edgecolor='k', s=75, alpha=0.9, label='Training data')
        ax.plot(x_line, y_line, color='#f59e0b', linewidth=3, label='Polynomial fit')
        ax.set_title('Polynomial regression fit')
        ax.set_xlabel('Score (%)')
        ax.set_ylabel('GPA')
        ax.legend()
        ax.grid(True, alpha=0.25)
        ax.set_xlim(40, 105)
        ax.set_ylim(0.8, 4.2)
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
    score = request.args.get('score', type=float)
    if score is None:
        return "Please provide a score parameter, e.g. /predict?score=85"
    predicted_gpa = predict_gpa(score)
    return f"Predicted GPA for score {score}%: {predicted_gpa:.2f}"

# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Scorely backend running → http://localhost:5000")
    app.run(debug=True, port=5000)
