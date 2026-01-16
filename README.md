# Student Performance Analysis Project

This project explores how student habits, engagement, and access factors relate to academic outcomes.
Using a dataset of **14,003 anonymized student records**, the goal is to understand:

- Which behaviors correlate most with strong academic performance
- How behavioral patterns group students into learning profiles
- Whether we can **predict FinalGrade categories** using machine learning

---

## Highlights
- Data cleaned and processed using reproducable steps
- **8 student behavior clusters** discovered using K-Means
- Model improved from **29% to 90% accuracy** through feature engineering
- Top drivers of performance identified (study efficiency, attendance, engagement)
- Next phase: **Interactive Streamlit prediction app**

---
Tech Stack
- Python 3.12
- pandas, numpy
- seaborn, matplotlib
- scikit-learn
    - Logistic Regression
    - K-Means
    - Random Forest
    - PCA
- Jupyter Notebook
- (Coming Soon) Streamlit for deployment

---

## Folder Structure

```student-performance-project/
│
├── data/
│ ├── cleaned_student_performance.csv
│ └── student_performance.csv
│
├── notebooks/
│ ├── 01_data_cleaning.ipynb
│ ├── 02_EDA.ipynb
│ ├── 03_clustering.ipynb
│ └── 04_modeling.ipynb
│
├── images/ # (plots/screenshots saved here)
│
├── requirements.txt
└── README.md```

---

### 1. Clean and Explore the Data
- Identify key variables
- Analyze relationships between behavior and FinalGrade
- Compare numeric + categorical patterns

### 2. Cluster student behavior
- Apply K-Means to find behavioral groups
- Examine differences in motivation, resources, and engagement

### 3. Train predictive models
- Baseline Logistic Regression (~29% accuracy)
- Random Forest Classifier (~90% accuracy)
- Evaluate performance using classification metrics & confusion matrix

### 4. Feature Engineering
Created new variables that turned out to be highly predictive:
- `StudyEfficiency`
- `AttendanceRatio`
- `TechAccess`
- `EngagementScore`
- `StressBalance`

### Next Steps
- Deploy model in a **Streamlit app**
- Add "what-if" behavior simulator
-Share insights visually on Instagram as part of an open learning journey

## Dataset Source
Student Performance & Learning Behavior Dataset
Public + CC BY 4.0
Kaggle - (link coming soon)

---

## Project Goals
Use data analytics + maching learning to answer:
> **What study behaviors and conditions most influence student success?**

---

This work sits at the intersection of:
- Data Science
- Learning Psychology
- Education equity

---

## How to run the project

Clone the repo: 

```bash
git clone <repo link coming soon>
cd student-performance-project

pip install -r requirements.txt


This is an ongoing progect - contributions, ideas, and feedback are welcome!