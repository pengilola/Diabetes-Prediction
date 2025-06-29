# Diabetes-Predictionfor Stark Health Clinic
A machine learning project predicting diabetes onset using patient health records for Stark Health Clinic.

# Project Overview
This project uses supervised machine learning techniques to predict diabetes onset based on patient data. It was developed as part of a data science initiative to improve early diagnosis and resource allocation for Stark Health Clinic.

## Problem Statement
Diabetes is a chronic disease with rising prevalence. Stark Health Clinic wants to predict patients likely to develop diabetes using structured health data and take preventive action.

## Dataset
- 100,000 patient records
- Features: Age, BMI, Blood Glucose, HbA1c, Smoking History, etc.
- Target: Diabetes diagnosis (1 = Yes, 0 = No)

## Tools Used
- Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
- Jupyter Notebook
- Models: Logistic Regression, Decision Tree, Random Forest

## Project Steps
1. Data Exploration & Cleaning
2. EDA (Univariate, Bivariate, Multivariate)
3. Feature Engineering
4. Model Training & Tuning (GridSearchCV)
5. Evaluation (Confusion Matrix, Precision, Recall)

## Best Model
- **Random Forest Classifier**  
- Accuracy: 97.2%  
- Strong predictors: HbA1c, Blood Glucose, Age, BMI

## File Overview
- `diabetes_prediction.ipynb`: Main analysis notebook
- `images/`: EDA & model visualizations 
- `requirements.txt`: Python libraries (optional)

## Next Steps
- Improve class imbalance
- Deploy model as web app using Streamlit or Flask

diabetes-prediction-project/
│
├── diabetes_prediction.ipynb               # Your main notebook
├── README.md                               # Your project description
├── diabetes_prediction_dataset.csv         # Your dataset
├── presentation_slides.pdf                 # Your final slides (optional, but nice)
├── project_requirements.pdf                # If required to show problem definition, etc.
└── images/                                 # Only if you have plots saved as images
     ├── plot1.png
     ├── heatmap.png


## Author
Ololade Folashade | June 2025
