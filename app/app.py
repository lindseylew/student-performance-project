import sys
from pathlib import Path

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier



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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features import add_engineered_features


# 3) Load data

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

if not DATA_PATH.exists():
    st.error(
        f"Data file not found: `{DATA_PATH}`\n\n"
        "Make sure `data/cleaned_student_performance.csv` exists."
    )
    st.stop()

data = load_data(DATA_PATH)

with st.sidebar:
    st.header("About the data")
    st.write(f"Rows: **{len(data):,}**")
    st.write("Target: **FinalGrade (0-3)**")
    st.caption("Encoding: 0=A (highest), 1=B, 2=C, 3=D (lowest)")
    st.divider()

# 4) Model training

FEATURE_COLS = [
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

@st.cache_resource
def train_model(data_path:str) -> RandomForestClassifier:
    df = pd.read_csv(data_path)
    df_fe = add_engineered_features(df)

    missing = [c for c in FEATURE_COLS + ["FinalGrade"] if c not in df_fe.columns]
    if missing:
        raise ValueError(f"Missing columns after feature engineering: {missing}")
    
    X = df_fe[FEATURE_COLS]
    y = df_fe["FinalGrade"]

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X, y)
    return rf

if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

st.subheader("Model status")

if not st.session_state.model_ready:
    st.warning("Model not trained yet for this session.")
    colA, colB = st.columns([1, 2], vertical_alignment="center")
    with colA:
        train_clicked = st.button("Train model", type="primary")
    with colB:
        st.caption("Deployment-safe: trains in memory")

    if train_clicked:
        try:
            with st.spinner("Training model..."):
                model = train_model(str(DATA_PATH))
            st.session_state.model_ready = True
            st.success("Model trained! You can now make predictions.")
            st.rerun()
        except Exception as e:
            st.exception(e)

    st.stop()

model = train_model(str(DATA_PATH))

# 5) Sidebar inputs (Predict Form)

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
        st.subheader("Study & Engagement")
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

# 6) Build input row + predict only on submit
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
X_input = input_fe[FEATURE_COLS]

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

# 7) Results UI

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