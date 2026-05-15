# Tech-Burnout-Checker


A machine learning project that predicts **burnout levels in tech workers** using workplace, lifestyle, and mental health features.  
Built with Python, scikit-learn, and XGBoost, this project explores multiple models and hypertuning to achieve state-of-the-art performance.

---

## 📌 Project Overview
Burnout is a growing challenge in the tech industry. This project leverages a Kaggle dataset on **Mental Health & Burnout in Tech Workers (2026)** to classify burnout severity into four categories:
- **Low**
- **Moderate**
- **High**
- **Severe**

The goal is to:
- Identify key predictors of burnout.
- Compare multiple machine learning models.
- Build a tuned, production-ready classifier.

---

##  Dataset
- Source: [Mental Health & Burnout in Tech Workers 2026](https://www.kaggle.com/datasets/mohankrishnathalla/mental-health-and-burnout-in-tech-workers-2026)
- Features include:
  - Demographics (age, gender, country, job role, seniority)
  - Work context (hours, meetings, team size, salary, company size)
  - Lifestyle (sleep, exercise, vacation, therapy access)
  - Psychosocial scores (stress, autonomy, manager support, job satisfaction)
  - Clinical scores (PHQ-9, GAD-7)

---

## ⚙️ Models Tested
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- AdaBoost  
- Support Vector Machine (SVM)  
- XGBoost  
- Naive Bayes  

---

## 🚀 Results
- **Gradient Boosting & XGBoost** → Achieved **100% accuracy and F1-score** across all burnout categories after hypertuning.  
- **Random Forest** → Nearly perfect performance, strong interpretability.  
- **Logistic Regression & SVM** → Excellent baselines (~98–99%).  
- **Naive Bayes** → Decent (~92%).  
- **AdaBoost** → Underperformed (~45%).  

---

## 🔧 Hypertuning
RandomizedSearchCV was used to optimize XGBoost parameters:
```python
Best Parameters: {
    'subsample': 1.0,
    'reg_lambda': 1,
    'reg_alpha': 1,
    'n_estimators': 100,
    'max_depth': 9,
    'learning_rate': 0.1,
    'gamma': 1,
    'colsample_bytree': 1.0
}
Best Score: 1.0
