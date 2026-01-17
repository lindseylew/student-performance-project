# Student Performance Analysis Project

This project explores how student habits, engagement, and access factors relate to academic outcomes.
Using a dataset of **14,003 anonymized student records**, the goal is to understand:

- Which behaviors correlate most with strong academic performance
- How behavioral patterns group students into learning profiles
- Whether we can **predict FinalGrade categories** using machine learning

---

## Project Highlights
- Data cleaned and processed using reproducable steps
- **8 student behavior clusters** discovered using K-Means
- Model improved from **29% to 90% accuracy** through feature engineering
- Key drivers of performance identified
    - Study efficiency
    - Attendance consistency
    - Engagement behaviors
- **Interactive Streamlit app** built to demonstrate real-time predictions

---

## Project Workflow

### 1️⃣ Data Cleaning & Exploration
    - Identified key behavioral and contextual variables
    - Analyzed relationships between behavior and FinalGrade
    - Compared numeric patterns and categorical patterns

### 2️⃣ Behavioal Clustering
    - Applied **K-Means clustering**
    - Identified distinct student learning profiles
    - Compared motivation, resources, and engagement across clusters

### 3️⃣ Predictive Modeling
    - Baseline **Logistic Regression** (~20% accuracy)
    - **Random Forest Classifier** (~90% accuracy)
    - Evaluated using:
        - Classification report
        - Confusion matrix
        - Class-level performance metrics

### 4️⃣ Feature Engineering
Created new variables that significantly improved performance:

- `StudyEfficiency`
- `AttendanceRatio`
- `TechAccess`
- `EngagementScore`
- `StressBalance`

Feature engineering logic is shared across noteboods and the Streamlit app for consistency

---

## Interactive Streamlit App

The project includes a Streamlit application that allows users to:

- Adjusts student behavior inputs
- Generate real-time grade predictions
- View class probabilities and confidence
- Inspect engineered features used by the model

> ⚠️ This app is for **educational demonstration only** and odes not make real academic decisions.

---

## Tech Stack

- **Python 3.12**
- pandas, numpy
- seaborn, matplotlib
- scikit-learn
    - Logistic Regression
    - K-Means
    - Random Forest
    - PCA
- Jupyter Notebook
- **Streamlit**

---

## Folder Structure

```text
student-performance-project/
│
├── app/
│   └── app.py                  # Streamlit application
│
├── data/
│   ├── cleaned_student_performance.csv
│   └── student_performance.csv
│
├── models/
│   └── random_forest_model.pkl # Trained model
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_modeling.ipynb
│
├── src/
│   └── features.py             # Shared feature engineering logic
│
├── images/                     # Saved plots / screenshots
│
├── requirements.txt
└── README.md
```

---

## How to run the project

Clone the repo: 

```bash
git clone git@github.com:lindseylew/student-performance-project.git
cd student-performance-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

This is an ongoing progect - contributions, ideas, and feedback are welcome!
