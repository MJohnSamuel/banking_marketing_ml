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
Logistic Regression achieved very high recall (0.9116), indicating strong capability in identifying customers who are likely to subscribe. However, precision is moderate, meaning the model predicts many positives, including false positives. Its strong AUC (0.9438) shows effective class separation. Logistic Regression performs well in imbalanced datasets when recall is prioritized.

### Decision Tree
Decision Tree achieved high accuracy (0.9144) but relatively lower AUC compared to ensemble models. While it captures non-linear relationships, it is more prone to overfitting. The moderate MCC score indicates reasonable but not optimal balanced classification performance.

### kNN
kNN achieved balanced performance but lower recall compared to Logistic Regression. Its performance depends heavily on feature scaling and distance metrics. The moderate AUC and MCC suggest that while kNN works reasonably well, it does not outperform ensemble methods on this dataset.

### Naive Bayes
Naive Bayes produced high recall (0.6907) but low precision (0.3495), indicating a high number of false positives. This behavior is expected due to its strong independence assumption among features. Although computationally efficient, its performance is limited when feature dependencies exist.

### Random Forest
Random Forest demonstrated strong overall performance with high AUC (0.9448) and good MCC (0.5146). As an ensemble method, it reduces overfitting compared to a single Decision Tree. It provides better generalization and more stable predictions across different splits.

### XGBoost
XGBoost achieved the highest AUC (0.9499) and strong MCC (0.5633), indicating the best overall balanced classification performance. It effectively handles non-linear relationships and class imbalance through boosting. Among all models, XGBoost provides the most reliable and consistent results.

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
