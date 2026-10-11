from pathlib import Path
import joblib

model_path = Path(__file__).with_name("student_pass_model.joblib")
model = joblib.load(model_path)

study_hours = [[2], [5], [7]]
predictions = model.predict(study_hours)

for hours, prediction in zip(study_hours, predictions):
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study Hours: {hours[0]} - Result: {result}")

print("\nThe saved model was loaded without retraining.")
