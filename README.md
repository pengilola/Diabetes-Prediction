# Diabetes Prediction for Stark Health Clinic

**Author:** Ololade Folashade  
**Date:** June 2025

## 📌 Project Overview
This project was developed for Stark Health Clinic to build a predictive model that identifies patients at risk of diabetes. By leveraging supervised machine learning techniques, this solution enables early detection, allowing for proactive intervention and optimized healthcare resource allocation.

## 🩺 Problem Statement
Diabetes poses serious health and economic burdens for patients and healthcare providers. Stark Health's current early detection methods lack precision. This project addresses that gap by developing an accurate, data-driven prediction model to flag high-risk patients using routine clinical data.

## Objectives
- Predict the likelihood of diabetes using patient health records.
- Evaluate and compare multiple classification models.
- Recommend a deployable solution for the clinic's preventive care system.

## Dataset
The dataset contains 100,000 anonymized patient records with the following features:
- `age`
- `gender`
- `hypertension`
- `heart_disease`
- `smoking_history`
- `bmi`
- `HbA1c_level`
- `blood_glucose_level`
- `diabetes` (target)

All data was cleaned and preprocessed before modeling. No missing values were present.

## Tools & Technologies
- Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)
- Jupyter Notebook
- Git/GitHub
- Anaconda
- Statistical modeling (OLS via Statsmodels)
- Machine learning models: Logistic Regression, Decision Tree, Random Forest

## Methodology
1. **Exploratory Data Analysis (EDA)**  
   Conducted univariate, bivariate, and multivariate analysis to understand feature distributions and relationships.

2. **Feature Engineering**  
   - One-hot encoding for categorical variables (`gender`, `smoking_history`)  
   - Feature scaling using `StandardScaler`  
   - Multicollinearity check using OLS regression

3. **Modeling**  
   - Trained and tuned:  
     - Logistic Regression  
     - Decision Tree  
     - Random Forest  
   - Model evaluation based on accuracy, precision, recall, and confusion matrix

4. **Results**  
   - Best model: **Random Forest** with ~97.2% accuracy  
   - Key predictors: HbA1c level, blood glucose, age, BMI

## Key Results
| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | ~95.9%   | 88% (pos) | 60%    | 72%      |
| Decision Tree       | ~97.2%   | 100% (pos)| 67%    | 81%      |
| Random Forest       | ~97.2%   | 100% (pos)| 68%    | 81%      |

> All models outperformed a naïve classifier that always predicts "no diabetes" (accuracy: ~91%).

## Conclusion
The final model enables Stark Health Clinic to:
- Identify at-risk patients with high confidence
- Reduce missed diagnoses
- Enable targeted follow-up care

## Next Steps
- Integrate the model into Stark Health's clinical workflow
- Deploy as a web app using Streamlit or Flask
- Improve performance with time-series data or real-time updates

## Folder Structure
diabetes-prediction-stark-health/
- diabetes_prediction.ipynb # Main notebook
- README.md # Project summary
- diabetes_prediction_dataset.csv # Dataset
- StarkHealth_Presentation.pdf
- Project_Brief.pdf
- image


## How to Run
1. Clone the repository  
2. Open `diabetes_prediction.ipynb` using Jupyter Notebook  
3. Run cells in order to view full analysis and results

> _This project was developed as part of a machine learning capstone under Full Stack Data Scientist Certificatiobn by 10alytics
