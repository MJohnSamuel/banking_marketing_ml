# Machine Learning Assignment 2  
## Bank Marketing Term Deposit Classification

---

## Problem Statement

The objective of this project is to build multiple machine learning classification models to predict whether a customer will subscribe to a term deposit based on banking marketing campaign data.

This project demonstrates:

- Implementation of six classification models
- Evaluation using multiple performance metrics
- Comparative analysis of model performance
- Deployment using Streamlit Community Cloud

---

## Dataset Description

**Dataset Name:** UCI Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  

The dataset contains marketing campaign information from a Portuguese banking institution. The goal is to predict whether a client subscribes to a term deposit.

### Dataset Characteristics:

- Total Instances: ~45,000+
- Number of Input Features: 16
- Target Variable: `y`
  - `yes` → Client subscribed
  - `no` → Client did not subscribe
- Problem Type: Binary Classification

The dataset is imbalanced, with a majority of "no" responses.

---

## Models Used

The following six classification models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

---

## Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.865380 | 0.943821 | 0.451682 | 0.911638 | 0.604070 | 0.581702 |
| Decision Tree | 0.914421 | 0.889335 | 0.649264 | 0.522629 | 0.579104 | 0.535972 |
| kNN | 0.907623 | 0.879891 | 0.625943 | 0.447198 | 0.521684 | 0.480309 |
| Naive Bayes | 0.820345 | 0.839329 | 0.349509 | 0.690733 | 0.464156 | 0.400918 |
| Random Forest | 0.915392 | 0.944778 | 0.691542 | 0.449353 | 0.544742 | 0.514567 |
| XGBoost | 0.917941 | 0.949892 | 0.659494 | 0.561422 | 0.606519 | 0.563309 |

---

## Observations

### Logistic Regression
Shows strong recall performance, making it suitable when minimizing false negatives is important. Handles class imbalance effectively.

### Decision Tree
Provides reasonable accuracy but may overfit compared to ensemble models.

### kNN
Moderate performance and sensitive to scaling and class imbalance.

### Naive Bayes
High recall but low precision, indicating many false positives.

### Random Forest
Strong ensemble performance with high AUC and improved generalization.

### XGBoost
Best overall balanced performance with highest AUC and MCC, making it the most reliable model among all six.

---

## Streamlit Application

The deployed Streamlit application includes:

- CSV dataset upload functionality
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix visualization

### Live App:
https://bankingmarketingml-ivsa9iqvy2hpjhgcwgxwpp.streamlit.app/

---

## GitHub Repository

https://github.com/MJohnSamuel/banking_marketing_ml
