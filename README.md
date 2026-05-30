# AI-Student-Burnout-Prediction
# 🎓 AI Student Burnout Prediction using XGBoost & SHAP

## 📌 Overview

This project explores the impact of AI-assisted learning on student well-being by predicting burnout risk using machine learning techniques.

The objective is to identify students who may be at risk of academic burnout based on demographic information, study habits, academic behavior, and AI usage patterns.

The project implements a complete machine learning workflow, including data preprocessing, feature engineering, model training, performance evaluation, and model interpretability through Explainable AI techniques.

---

## 🚀 Project Highlights

✅ Data preprocessing and cleaning

✅ Categorical feature encoding

✅ Binary classification problem formulation

✅ Handling class imbalance

✅ XGBoost model training

✅ ROC-AUC performance evaluation

✅ Feature importance analysis

✅ Explainable AI using SHAP

---

## 📂 Dataset

This project uses the publicly available dataset:

### Impact of AI on Students Dataset

Source:

https://www.kaggle.com/datasets/arafangsanneh/impact-of-ai-on-students

### Dataset Description

The dataset contains information related to:

- Student demographics
- Academic performance
- Study habits
- AI tool usage behavior
- Learning patterns
- Burnout risk indicators

The goal is to investigate whether AI-related behaviors can help predict student burnout risk.

---

## 🔍 Problem Statement

Student burnout has become an increasingly important concern in modern education.

As AI tools become integrated into learning environments, understanding their relationship with student well-being is essential.

This project aims to build an interpretable machine learning model capable of identifying students who may be at risk of burnout based on their academic and AI-assisted learning behaviors.

---

## ⚙️ Project Workflow

### 1. Data Exploration

The dataset is explored to understand:

- Feature distributions
- Data types
- Class distribution
- Potential data quality issues

Visualization techniques are used to inspect the target variable and identify class imbalance.

---

### 2. Data Preprocessing

The following preprocessing steps are applied:

- Removal of non-informative identifiers
- Encoding categorical variables using Label Encoding
- Preparation of model-ready features
- Creation of target labels

---

### 3. Feature Engineering

The original burnout categories are transformed into a binary target variable:

```python
Burnout_Risk_Binary
```

This converts the task into a supervised binary classification problem.

---

### 4. Train-Test Split

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

to evaluate model generalization on unseen samples.

---

### 5. Handling Class Imbalance

To improve learning on underrepresented classes, class imbalance is addressed using:

```python
scale_pos_weight
```

This helps the model better capture minority class patterns.

---

## 🤖 Machine Learning Model

### XGBoost Classifier

The primary model used in this project is:

```python
XGBClassifier(
    n_estimators=700,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8
)
```

### Why XGBoost?

XGBoost was selected because it:

- Performs exceptionally well on structured/tabular datasets
- Handles nonlinear relationships effectively
- Provides feature importance scores
- Offers strong generalization capabilities
- Supports class imbalance adjustments

---

## 📊 Model Evaluation

Model performance is assessed using multiple evaluation metrics:

### Accuracy

Measures overall prediction correctness.

### Precision

Measures how many positive predictions are actually correct.

### Recall

Measures how many actual positive cases are successfully identified.

### F1-Score

Balances Precision and Recall.

### ROC-AUC Score

Evaluates the model's ability to distinguish between classes across different classification thresholds.

---

## 📈 ROC Curve Analysis

A Receiver Operating Characteristic (ROC) curve is generated to visualize classification performance.

The ROC-AUC metric provides a threshold-independent measure of model quality and discriminative power.

---

## 📉 Feature Importance

Feature importance scores are extracted from the trained XGBoost model to identify which variables contribute most to burnout prediction.

This analysis helps answer questions such as:

- Which behaviors are most associated with burnout?
- Which factors have the strongest predictive power?
- How does AI usage influence model decisions?

---

## 🔬 Explainable AI (XAI)

One of the most important parts of this project is model interpretability.

The project uses SHAP (SHapley Additive exPlanations) to explain model predictions.

### SHAP Benefits

- Global feature importance analysis
- Local prediction explanations
- Transparent decision-making
- Improved trust in machine learning models

Implementation:

```python
explainer = shap.TreeExplainer(model)
```

Visualization:

```python
shap.summary_plot(...)
```

SHAP allows us to understand how each feature contributes to increasing or decreasing burnout risk.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- SHAP

---

## 📁 Project Structure

```text
AI-Student-Burnout-Prediction/
│
├── firstml.ipynb
├── README.md
├── requirements.txt
│
├── data/
│   └── impact_of_ai_on_students.csv
│
└── results/
    ├── roc_curve.png
    ├── feature_importance.png
    └── shap_summary.png
```

---

## 📊 Results

The project demonstrates how machine learning can be used to identify burnout risk among students using academic and AI-related behavioral data.

Key outcomes include:

- Successful burnout risk prediction
- Identification of influential factors
- Handling of class imbalance
- Interpretable predictions through SHAP
- End-to-end machine learning pipeline implementation

---

## 🚀 Future Improvements

Potential future enhancements include:

- Hyperparameter optimization using Optuna
- Cross-validation experiments
- Comparison with Random Forest and LightGBM
- Multi-class burnout prediction
- Deep Learning approaches
- Interactive dashboard deployment
- Model deployment using Streamlit or Flask

---

## 👩‍💻 Author

**Elena Ahouei**

---

## ⭐ If you find this project useful

Consider giving the repository a star and sharing feedback.
