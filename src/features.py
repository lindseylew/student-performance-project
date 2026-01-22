import pandas as pd
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    num_cols = [
        "StudyHours", "Attendance", "AssignmentCompletion", "OnlineCourses", "Internet", "Resources", "Discussions", "Extracurricular", "Motivation", "StressLevel"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[num_cols] = df[num_cols].fillna(0)

    df["StudyEfficiency"] = df['StudyHours'] / (df['AssignmentCompletion'] + 1)
    df["AttendanceRatio"] = df['Attendance'] / 100
    df['TechAccess'] = df['Internet'] + df['Resources']
    df['EngagementScore'] = df['Discussions'] + df['OnlineCourses'] + df['Extracurricular']
    df['StressBalance'] = df['Motivation'] - df['StressLevel']

    return df