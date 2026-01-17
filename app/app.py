import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

# 1) Page config

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")
st.write(
    "Interactive demo based on my student performance analysis project.\n\n"
    "This app uses a **Random Forest** model to predict **FinalGrade (0-3)** from study habits, engagement, and context."
)

st.info("Note: In this dataset, **FinalGrade is encoded as 0=highest (A) and 3=lowest (D)")

grade_label= {
    0: "A (highest)",
    1: "B",
    2: "C",
    3: "D (lowest)"
}

# 2) Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_student_performance.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"

import sys
sys.path.append(str(PROJECT_ROOT))

from src.features import add_engineered_features

# 3) Load data + model

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

data = load_data()
model = load_model()

st.sidebar.header("About the data")
st.sidebar.write(f"Rows: **{len(data):,}**")
st.sidebar.write("Target: **FinalGrade (0-3)**")
st.sidebar.caption("Encoding: 0=A (highest), 1=B, 2=C, 3=D (lowest)")

#4) Feature columns

feature_cols = [
    "StudyHours",
    "Attendance",
    "AssignmentCompletion",
    "OnlineCourses",
    "Motivation",
    "Resources",
    "Internet",
    "Discussions",
    "StressLevel",
    "Extracurricular",
    "StudyEfficiency",
    "AttendanceRatio",
    "TechAccess",
    "EngagementScore",
    "StressBalance"
]

# 5) Sidebar Inputs

st.sidebar.header("Input student features")

study_hours = st.sidebar.slider("StudyHours", 0, 20, 5)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 85)
assignment_completion = st.sidebar.slider("AssignmentCompletion (%)", 0, 100, 80)
online_courses = st.sidebar.slider("OnlineCourses (#)", 0, 10, 1)

motivation = st.sidebar.selectbox("Motivation (0=Low, 1=Medium, 2=High)", [0, 1, 2], index=1)
resources = st.sidebar.selectbox("Resources (0=Low, 1=Medium, 2=High)", [0, 1, 2], index=1)

internet = st.sidebar.selectbox("Internet (0=No, 1=Yes)", [0, 1], index=1)
discussions = st.sidebar.selectbox("Discussions (0=No, 1=Yes)", [0, 1], index=1)

stress_level = st.sidebar.slider("StressLevel (0-10)", 0, 10, 5)
extracurricular = st.sidebar.selectbox("Extracurricular (0=No, 1=Yes)", [0, 1], index=0)

# single-row dataframe (raw inputs)

input_df = pd.DataFrame([{
    "StudyHours": study_hours,
    "Attendance": attendance,
    "AssignmentCompletion": assignment_completion,
    "OnlineCourses": online_courses,
    "Motivation": motivation,
    "Resources": resources,
    "Internet": internet,
    "Discussions": discussions,
    "StressLevel": stress_level,
    "Extracurricular": extracurricular,
}])

# Apply feature engineering
input_fe = add_engineered_features(input_df)

# Ensure that we only pass the columns the model expects, in the correct order
X_input = input_fe[feature_cols]

# 6) Predict

st.subheader("Prediction")

pred = model.predict(X_input)[0]

# Some models may support predict_proba

proba = None
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X_input)[0]

st.metric("Predicted FinalGrade", f"{pred} -> {grade_label[pred]}")

if proba is not None:
    st.write("**Confidence (class probabilities):**")
    proba_df = pd.DataFrame({
        "FinalGrade": [0, 1, 2, 3],
        "Label": [grade_label[i] for i in [0, 1, 2, 3]],
        "Probability": proba
    })
    st.dataframe(proba_df, use_container_width=True)

st.markdown("###What might help improve this outcome?")
st.write("Based on the model's learned patterns, improvments in **study efficiency**"
         "and **engagement** tend to have the largest impact on predicted outcomes."
        )

st.divider()

# 7) Show engineered features (transparency)

with st.expander("See engineered features used by the model"):
    st.write("These features are computed from the inputs (same logic as the modeling notebook).")
    engineered_cols = ["StudyEfficiency", "AttendanceRatio", "TechAccess", "EngagementScore", "StressBalance"]
    show_df = input_fe.copy()
    st.dataframe(show_df[engineered_cols], use_container_width=True)

# 8) Interpretation Text

st.subheader("How to read this result")
st.write(
    "- This is a **demo** trained on a public dataset.\n"
    "- The model learns patterns fro **behavior and context**, but real student outcomes are also shaped by factors not in the data.\n"
    "- Predictions are **probabilistic**, not guarantees."
)

st.caption("⚠️ This tool is for educational demonstration only and does not make real academic decisions.")