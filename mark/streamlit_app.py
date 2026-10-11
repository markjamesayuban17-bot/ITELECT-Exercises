from pathlib import Path
import joblib
import streamlit as st
import matplotlib.pyplot as plt

model_path = Path(__file__).with_name("student_pass_model.joblib")
model = joblib.load(model_path)

st.set_page_config(page_title="Student Result Predictor", page_icon="🎓")
st.title("Student Result Prediction System")
st.write("Predict a student's result based on daily study hours.")

with st.form("prediction_form"):
    student_name = st.text_input("Student Name")
    study_hours = st.number_input(
        "Daily Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=1.0
    )
    submitted = st.form_submit_button("Predict Result")

if submitted:
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = model.predict([[study_hours]])[0]
        result = "Pass" if prediction == 1 else "Fail"

        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours}")

        if prediction == 1:
            st.success(f"Predicted Result: {result}")
        else:
            st.error(f"Predicted Result: {result}")

        st.info(
            "The saved model was loaded and given the study hours as input. "
            "It learned from the example training data. This prediction is "
            "an estimate, not a guaranteed academic result."
        )

        fig, ax = plt.subplots()
        ax.bar(["Daily Study Hours"], [study_hours])
        ax.set_ylabel("Hours")
        ax.set_ylim(0, 24)
        ax.set_title(f"Study Hours - {student_name}")
        st.pyplot(fig)
        plt.close(fig)

st.caption("Student Result Prediction System")
