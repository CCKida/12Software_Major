from flask import Flask, render_template_string, request, jsonify, send_from_directory
import os
import json
import datetime

app = Flask(__name__, static_folder="static", template_folder="templates")

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

# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Scorely backend running → http://localhost:5000")
    app.run(debug=True, port=5000)
