from pathlib import Path
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

model_path = Path(__file__).with_name("student_pass_model.joblib")
model = joblib.load(model_path)


@app.route("/")
def home():
    return "Student Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please provide valid JSON data"}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide numeric study_hours and attendance_percentage"
        }), 400

    if not 0 <= study_hours <= 24:
        return jsonify({"error": "Study hours must be between 0 and 24"}), 400

    if not 0 <= attendance <= 100:
        return jsonify({"error": "Attendance must be between 0 and 100"}), 400

    # This example model predicts using study hours only.
    prediction = model.predict([[study_hours]])[0]
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)
