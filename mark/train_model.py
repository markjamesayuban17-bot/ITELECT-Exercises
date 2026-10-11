from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression

# Example training data: study hours and result (0 = Fail, 1 = Pass)
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

model = LogisticRegression()
model.fit(X, y)

model_path = Path(__file__).with_name("student_pass_model.joblib")
joblib.dump(model, model_path)

print("Model trained and saved successfully!")
print(f"Saved file: {model_path.name}")
