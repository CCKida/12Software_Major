# Scorely

Scorely is a Flask-powered website for tracking student performance, calculating GPA, and reflecting on academic progress. It includes a polished static frontend, a GPA prediction model using polynomial regression, and an admin dashboard for visualizing training data.

## Features

- Home page with project overview and navigation
- GPA calculator page for entering subjects, scores, and unit values
- Reflection form page for personal progress notes and mood tracking
- Admin dashboard with machine learning charts for training data and polynomial regression fit
- Predictive endpoint for estimating unit requirement from a given score
- Data persistence via `user_gpa_data.csv`

## Pages

- `/` - Home page (`index.html`)
- `/gpa` - GPA calculator page (`gpa.html`)
- `/reflect` - Reflection form page (`reflect.html`)
- `/admin` - Admin dashboard page (`admin.html`)
- `/admin/plot/score-vs-gpa` - Training data visualization
- `/admin/plot/polynomial-fit` - Polynomial regression visualization
- `/predict?grade=<score>` - Predict unit value for a score

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

- `data.csv` - Base training dataset used for polynomial regression
- `user_gpa_data.csv` - Persisted user GPA submissions and subject records
- `reflect_training.csv` / `reflect_predictions.csv` - Related reflection data files (used by reflection workflow)

## Notes

- The app automatically retrains the regression model after each GPA submission.
- The `/log_gpa` endpoint saves submitted GPA and subject data to `user_gpa_data.csv`.
- If `user_gpa_data.csv` does not exist, it is created automatically on first submission.

