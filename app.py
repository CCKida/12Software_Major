from flask import Flask, render_template_string, request, jsonify, send_from_directory
import os
import json
import datetime

app = Flask(__name__, static_folder="static", template_folder="templates")

# ── In-memory store (replace with a real DB in production) ──────────────────
reflections: list[dict] = []


# ── Helper: calculate GPA ────────────────────────────────────────────────────
GRADE_POINTS = {
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "D-": 0.7,
    "F":  0.0,
}

def score_to_grade(score: float) -> str:
    """Convert a 0-100 percentage score to a letter grade."""
    if score >= 93: return "A+"
    if score >= 90: return "A"
    if score >= 87: return "A-"
    if score >= 83: return "B+"
    if score >= 80: return "B"
    if score >= 77: return "B-"
    if score >= 73: return "C+"
    if score >= 70: return "C"
    if score >= 67: return "C-"
    if score >= 63: return "D+"
    if score >= 60: return "D"
    if score >= 57: return "D-"
    return "F"

def calculate_gpa(subjects: list[dict]) -> dict:
    """
    Calculate GPA from a list of subjects.

    Each subject dict should contain:
      - name  (str)
      - score (float, 0-100)   OR
      - grade (str, e.g. 'A-')
      - units (float, credit hours — defaults to 1)

    Returns a summary dict.
    """
    if not subjects:
        return {"error": "No subjects provided."}

    total_points = 0.0
    total_units  = 0.0
    breakdown    = []

    for s in subjects:
        name  = s.get("name", "Unnamed")
        units = float(s.get("units", 1))

        if "grade" in s:
            letter = s["grade"].upper().strip()
        elif "score" in s:
            letter = score_to_grade(float(s["score"]))
        else:
            return {"error": f"Subject '{name}' has neither a grade nor a score."}

        points = GRADE_POINTS.get(letter)
        if points is None:
            return {"error": f"Unrecognised grade '{letter}' for subject '{name}'."}

        weighted = points * units
        total_points += weighted
        total_units  += units

        breakdown.append({
            "name":        name,
            "grade":       letter,
            "grade_point": points,
            "units":       units,
        })

    gpa = round(total_points / total_units, 2) if total_units else 0.0

    return {
        "gpa":          gpa,
        "total_units":  total_units,
        "breakdown":    breakdown,
        "classification": _classify(gpa),
    }

def _classify(gpa: float) -> str:
    if gpa >= 3.7: return "High Distinction"
    if gpa >= 3.3: return "Distinction"
    if gpa >= 3.0: return "Credit"
    if gpa >= 2.0: return "Pass"
    return "Below Pass"


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve the static homepage."""
    return send_from_directory(".", "index.html")


@app.route("/api/gpa", methods=["POST"])
def api_gpa():
    """
    POST /api/gpa
    Body (JSON):
      {
        "subjects": [
          {"name": "Maths",   "score": 88, "units": 3},
          {"name": "English", "grade": "A-", "units": 3},
          {"name": "Science", "score": 74}
        ]
      }
    """
    data = request.get_json(force=True)
    subjects = data.get("subjects", [])
    result = calculate_gpa(subjects)
    status = 400 if "error" in result else 200
    return jsonify(result), status


@app.route("/api/reflect", methods=["POST"])
def api_reflect():
    """
    POST /api/reflect
    Body (JSON):
      {
        "student_name":  "Alex",
        "subject":       "Mathematics",
        "assessment":    "Mid-year exam",
        "score":         72,
        "what_went_well": "...",
        "what_to_improve": "...",
        "action_plan":    "..."
      }
    """
    data = request.get_json(force=True)

    required = ["subject", "assessment", "what_went_well", "what_to_improve", "action_plan"]
    missing  = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    entry = {
        "id":              len(reflections) + 1,
        "timestamp":       datetime.datetime.utcnow().isoformat() + "Z",
        "student_name":    data.get("student_name", "Anonymous"),
        "subject":         data["subject"],
        "assessment":      data["assessment"],
        "score":           data.get("score"),
        "what_went_well":  data["what_went_well"],
        "what_to_improve": data["what_to_improve"],
        "action_plan":     data["action_plan"],
    }
    reflections.append(entry)
    return jsonify({"message": "Reflection saved.", "entry": entry}), 201


@app.route("/api/entries", methods=["GET"])
def api_entries():
    """
    GET /api/entries?student=Alex
    Returns all (or filtered) reflection entries.
    """
    student = request.args.get("student", "").strip().lower()
    if student:
        filtered = [r for r in reflections if r["student_name"].lower() == student]
    else:
        filtered = reflections
    return jsonify({"count": len(filtered), "entries": filtered})


@app.route("/api/summary", methods=["GET"])
def api_summary():
    """
    GET /api/summary?student=Alex
    Returns GPA summary across all reflection entries that have a score.
    """
    student = request.args.get("student", "").strip().lower()
    pool = [r for r in reflections if r.get("score") is not None]
    if student:
        pool = [r for r in pool if r["student_name"].lower() == student]

    if not pool:
        return jsonify({"message": "No scored entries found."}), 404

    subjects = [{"name": r["subject"], "score": r["score"]} for r in pool]
    result = calculate_gpa(subjects)
    result["entries_used"] = len(pool)
    return jsonify(result)


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Scorely backend running → http://localhost:5000")
    app.run(debug=True, port=5000)
