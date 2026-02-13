import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="Bank Marketing ML App")

st.title("Bank Marketing Term Deposit Prediction")
st.write("Upload test dataset and evaluate selected model.")

# ------------------------------
# Model selection
# ------------------------------
model_files = {
    "Logistic Regression": "model/Logistic_Regression.joblib",
    "Decision Tree": "model/Decision_Tree.joblib",
    "kNN": "model/kNN.joblib",
    "Naive Bayes": "model/Naive_Bayes.joblib",
    "Random Forest": "model/Random_Forest.joblib",
    "XGBoost": "model/XGBoost.joblib"
}

selected_model_name = st.selectbox(
    "Select Model",
    list(model_files.keys())
)

# ------------------------------
# File upload
# ------------------------------
uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        test_df = pd.read_csv(uploaded_file, sep=';')
    except:
        st.error("Error reading file. Ensure correct format.")
        st.stop()

    if "y" not in test_df.columns:
        st.error("Uploaded CSV must contain target column 'y'.")
        st.stop()

    X_test = test_df.drop("y", axis=1)
    if test_df["y"].dtype == object:
       y_test = (
        test_df["y"]
        .str.strip()
        .map({"no": 0, "yes": 1})
    )
    else:
       y_test = test_df["y"]

    if y_test.isna().any():
      st.error("Target column 'y' contains unexpected values.")
      st.stop()

    model_path = model_files[selected_model_name]

    if not os.path.exists(model_path):
        st.error("Model file not found.")
        st.stop()

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ------------------------------
    # Metrics
    # ------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"AUC Score: {auc:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"MCC Score: {mcc:.4f}")

    # ------------------------------
    # Confusion Matrix
    # ------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.matshow(cm)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)