import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

st.title("ML- Assignment II- Heart Disease Classification – Multiple Models")

st.write("Upload a CSV in the same format as the heart disease dataset to evaluate different classifiers.")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of uploaded data:")
    st.dataframe(df.head())

    target_col = st.selectbox("Select target column", df.columns, index=len(df.columns) - 1)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_name = st.selectbox(
            "Select model",
            ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"]
        )

        if st.button("Train and Evaluate"):
            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train_scaled, y_train)
                X_eval = X_test_scaled
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
                model.fit(X_train, y_train)
                X_eval = X_test
            elif model_name == "kNN":
                model = KNeighborsClassifier(n_neighbors=5)
                model.fit(X_train_scaled, y_train)
                X_eval = X_test_scaled
            elif model_name == "Naive Bayes":
                model = GaussianNB()
                model.fit(X_train, y_train)
                X_eval = X_test
            elif model_name == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                X_eval = X_test
            else:  # XGBoost
                model = XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss"
                )
                model.fit(X_train, y_train)
                X_eval = X_test

            y_pred = model.predict(X_eval)

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_eval)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            else:
                auc = np.nan

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)

            st.subheader("Evaluation Metrics")
            st.write(f"Accuracy: {acc:.3f}")
            st.write(f"AUC: {auc:.3f}")
            st.write(f"Precision: {prec:.3f}")
            st.write(f"Recall: {rec:.3f}")
            st.write(f"F1 Score: {f1:.3f}")
            st.write(f"MCC: {mcc:.3f}")

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred, zero_division=0))
else:
    st.info("Please upload a CSV file to begin.")
