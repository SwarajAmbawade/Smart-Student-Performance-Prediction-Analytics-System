import os
import csv
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

CSV_FILE = "predictions_log.csv"

# Ensure CSV exists with headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "attendance",
            "marks",
            "study_hours",
            "probability",
            "status",
            "grade"
        ])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    attendance = float(request.form["attendance"])
    marks = float(request.form["marks"])
    hours = float(request.form["hours"])

    features = np.array([[attendance, marks, hours]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction] * 100

    if prediction == 1:
        status = "PASS"
        if probability >= 90:
            grade = "A"
            level = "Excellent Performance"
        elif probability >= 75:
            grade = "B"
            level = "Good Performance"
        else:
            grade = "C"
            level = "Average Performance"

        suggestion = "Maintain consistency and continue effective study habits."
        color = "green"
    else:
        status = "FAIL"
        grade = "D"
        level = "At Academic Risk"
        suggestion = "Increase study hours and improve attendance urgently."
        color = "red"

    # Log prediction
    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            attendance,
            marks,
            hours,
            round(probability, 2),
            status,
            grade
        ])

    return render_template(
        "result.html",
        status=status,
        probability=round(probability, 2),
        level=level,
        grade=grade,
        suggestion=suggestion,
        color=color
    )

@app.route("/dashboard")
def dashboard():
    grades = {"A": 0, "B": 0, "C": 0, "D": 0}
    status = {"PASS": 0, "FAIL": 0}
    attendance = []
    probability = []

    with open(CSV_FILE, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            grades[row["grade"]] += 1
            status[row["status"]] += 1
            attendance.append(float(row["attendance"]))
            probability.append(float(row["probability"]))

    return render_template(
        "dashboard.html",
        grades=grades,
        status=status,
        attendance=attendance,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)
