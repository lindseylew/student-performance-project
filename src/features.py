import pandas as pd
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["StudyEfficiency"] = df['StudyHours'] / (df['AssignmentCompletion'] + 1)
    df["AttendanceRatio"] = df['Attendance'] / 100
    df['TechAccess'] = df['Internet'] + df['Resources']
    df['EngagementScore'] = df['Discussions'] + df['OnlineCourses'] + df['Extracurricular']
    df['StressBalance'] = df['Motivation'] - df['StressLevel']

    return df