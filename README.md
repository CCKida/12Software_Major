# Scorely

Scorely is a Flask-powered website for tracking student performance, calculating GPA, and reflecting on academic progress. It includes a static frontend, a polynomial regression prediction model, and SQLite-backed persistence for GPA and reflection data.

Made by students for students. 

## Features

- Static home page with project overview and navigation
- GPA calculator page for entering subjects, scores, difficulty, and unit values
- Reflection page for viewing and exporting reflection prediction data
- Predictive endpoint for estimating unit requirements based on grade and difficulty
- SQLite persistence for GPA entries and reflection records
- CSV migration support from legacy `user_gpa_data.csv`

## Endpoints

- `/` - Home page (`index.html`)
- `/gpa` - GPA calculator page (`gpa.html`)
- `/reflect` - Reflection form page (`reflect.html`)
- `/predict?grade=<score>&difficulty=<difficulty>` - Predict unit value for a grade
- `/log_gpa` - POST endpoint to save GPA submissions and retrain the model
- `/reflect_predictions.csv` - Export reflection prediction CSV

## Requirements

- Python 3.8+ (recommended)
- Flask
- pandas
- numpy
- scikit-learn
- matplotlib

## Setup

1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install flask pandas numpy scikit-learn matplotlib
```

3. Run the application:

```powershell
python app.py
```

4. Open your browser and go to:

```text
http://localhost:5000
```

## Data files

- `data.csv` - Base training dataset used to build the polynomial regression model
- `user_gpa_data.csv` - Legacy GPA submission CSV used for migration into SQLite
- `gpa_data.db` - SQLite database storing GPA submissions and subject records
- `reflection_data.db` - SQLite database storing reflection training and prediction records
- `reflect_predictions.csv` - Exportable reflection prediction CSV endpoint

## Notes

- `app.py` initializes SQLite databases automatically if needed and migrates legacy `user_gpa_data.csv` entries into `gpa_data.db`.
- The GPA model retrains after every `/log_gpa` POST submission.
- `/predict` accepts `grade` and optional `difficulty`; difficulty is normalized and mapped to numeric weight values.

## Deployment

Minimal steps to run in production (Linux container / Heroku):

1. Install pinned dependencies:

```bash
pip install -r requirements.txt
```

2. Run with Gunicorn:

```bash
gunicorn wsgi:application --bind 0.0.0.0:8000
```

On platforms like Heroku the provided `Procfile` will run Gunicorn automatically and the `PORT` environment variable will be honored.
