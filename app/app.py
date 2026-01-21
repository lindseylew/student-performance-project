import sys
from pathlib import Path

import joblib
import streamlit as st
import pandas as pd



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

st.info("Note: In this dataset, **FinalGrade is encoded as 0=highest (A) and 3=lowest (D)**")

grade_label= {
    0: "A (highest)",
    1: "B",
    2: "C",
    3: "D (lowest)"
}

# 2) Paths + imports from src

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_student_performance.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features import add_engineered_features

# 3) Load data + model

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_model_or_none(path: Path):
    """Load a trained model if present. Return None if missing."""
    if path.exists():
        return joblib.load(path)
    return None

# Show a clear error if data is missing
if not DATA_PATH.exists():
    st.error(f"Data file not found: `{DATA_PATH}`\n\Make sure `data/cleaned_student_performance.csv` exists.")
    st.stop()

data = load_data(DATA_PATH)
model = load_model_or_none(MODEL_PATH)

with st.sidebar:
    st.sidebar.header("About the data")
    st.sidebar.write(f"Rows: **{len(data):,}**")
    st.sidebar.write("Target: **FinalGrade (0-3)**")
    st.sidebar.caption("Encoding: 0=A (highest), 1=B, 2=C, 3=D (lowest)")
    st.divider()

# 4) Missing model UX

if model is None:
    st.warning(
        "Trained model file not found (`models/random_forest.pkl`).\n\n"
        "This repo does not include model artifacts (GitHub file size limit)."
    )

    with st.expander("Train a demo model locally (recommended for testing)"):
        st.write(
            "Click to train a Random Forest using the cleaned dataset.\n\n"
            "**Note:** This is for local/demo use only."
            )

        if st.button("Train demo model"):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split

            data_fe = add_engineered_features(data)

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
                "StressBalance",
            ]

            X = data_fe[feature_cols]
            y = data_fe["FinalGrade"]

            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            rf = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            )
            rf.fit(X_train,y_train)

            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(rf, MODEL_PATH)

            st.success("Model trained and saved to `models/random_forest.pkl`. Reloading app...")
            st.rerun()
    st.stop()

#5) Feature columns

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

# 6) Sidebar Inputs

defaults = {
    "StudyHours": 5,
    "Attendance": 85,
    "AssignmentCompletion": 80,
    "OnlineCourses": 1,
    "Motivation": 1,
    "Resources": 1,
    "Internet": 1,
    "Discussions": 1,
    "StressLevel": 5,
    "Extracurricular": 0,
}

with st.sidebar:
    st.header("Input student features")


    with st.form("predict_form"):
        st.subheader("Study & Engagment")
        study_hours = st.slider("StudyHours", 0, 20, defaults["StudyHours"], help="Weekly study time (0-20).")
        attendance = st.slider("Attendance (%)", 0, 100, defaults["Attendance"])
        assignment_completion = st.slider("AssignmentCompletion (%)", 0, 100, defaults["AssignmentCompletion"])
        online_courses = st.slider("OnlineCourses (#)", 0, 10, defaults["OnlineCourses"])

        st.subheader("Context")
        motivation = st.selectbox(
            "Motivation",
            [0, 1, 2], 
            index=defaults["Motivation"],
            help="Encoded: 0=Low, 1=Medium, 2=High",
            )
        resources = st.selectbox(
            "Resources", 
            [0, 1, 2], 
            index=defaults["Resources"],
            help="Encoded: 0=Low, 1=Medium, 2=High",
        )
        internet = st.selectbox("Internet access", [0, 1], index=defaults["Internet"], help="0=No, 1=Yes")
        discussions = st.selectbox("Participates in discussions", [0, 1], index=defaults["Discussions"], help="0=No, 1=Yes")
        extracurricular = st.selectbox("Extracurricular", [0, 1], index=defaults["Extracurricular"], help="0=No, 1=Yes")
        stress_level = st.slider("StressLevel (0-10)", 0, 10, defaults["StressLevel"])

        predict_clicked = st.form_submit_button("Predict")

# 7) Build input row + predict only on submit

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

if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
    st.session_state.last_proba = None
    st.session_state.last_engineered = None

if predict_clicked:
    pred = int(model.predict(X_input)[0])

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0].tolist()

    st.session_state.last_pred = pred
    st.session_state.last_proba = proba
    st.session_state.last_engineered = input_fe.copy()

# 8) Results UI

st.subheader("Prediction")

if st.session_state.last_pred is None:
    st.info("Set your inputs in the sidebar, then click **Predict**.")
    st.stop()

pred = st.session_state.last_pred
proba = st.session_state.last_proba

st.metric("Predicted Final Grade", f"{pred} → {grade_label[pred]}")

if proba is not None:
    st.write("**Confidence (class probabilities):**")
    proba_df = pd.DataFrame({
        "FinalGrade": [0, 1, 2, 3],
        "Label": [grade_label[i] for i in [0, 1, 2, 3]],
        "Probability": proba,
    }).sort_values("FinalGrade")
    st.dataframe(proba_df, width="stretch", hide_index=True)

st.markdown("### What might help improve this outcome?")
st.write(
    "In this dataset, the strongest drivers of predicted outcomes were **study efficiency**, "
    "**assignment completion**, **attendance**, and **engagement**.\n\n"
    "Try adjusting those inputs and rerun **Predict** to explore a few what-if scenarios."
)

st.divider()

with st.expander("Show engineered features used by the model"):
    st.write("These are computed from your inputs using the same logic as the modeling notebook.")
    engineered_cols = ["StudyEfficiency", "AttendanceRatio", "TechAccess", "EngagementScore", "StressBalance"]
    show_df = st.session_state.last_engineered[engineered_cols].copy()
    st.dataframe(show_df, width="stretch", hide_index=True)

st.subheader("How to read this result")
st.write(
    "- This is a **demo** trained on a public dataset.\n"
    "- The model learns patterns from **behavior and context**, but real outcomes are shaped by many factors not captured here.\n"
    "- Predictions are **probabilistic**, not guarantees."
)

st.caption("⚠️ For educational demonstration only. Not for real academic decision-making.")
