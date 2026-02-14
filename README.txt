# Heart Disease Classification – Assignment 2

## Problem Statement

The goal of this project is to build and compare multiple machine learning classifiers to predict the presence of heart disease in patients based on their clinical and demographic attributes. The task is framed as a supervised binary classification problem where the target variable indicates whether a patient is diagnosed with heart disease (1) or not (0). The project follows an end‑to‑end workflow: data loading and preprocessing, model training and evaluation, and deployment of an interactive Streamlit web application for demonstration.

## Dataset Description

The dataset used is a heart disease dataset obtained from a public repository (Kaggle/UCI style tabular data). Each row corresponds to one patient, and each column represents a clinical measurement or risk factor.

- **Number of instances (rows):** > 500 (the uploaded heart.csv file)  
- **Number of attributes:** 13 input features + 1 target label (`target`)  

**Input Features**

- `age` – Age of the patient (in years)  
- `sex` – Sex (1 = male, 0 = female)  
- `cp` – Chest pain type (categorical encoded as 0–3)  
- `trestbps` – Resting blood pressure (in mm Hg)  
- `chol` – Serum cholesterol (in mg/dl)  
- `fbs` – Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)  
- `restecg` – Resting electrocardiographic results (encoded)  
- `thalach` – Maximum heart rate achieved  
- `exang` – Exercise induced angina (1 = yes, 0 = no)  
- `oldpeak` – ST depression induced by exercise relative to rest  
- `slope` – Slope of the peak exercise ST segment  
- `ca` – Number of major vessels (0–3) colored by fluoroscopy  
- `thal` – Thalassemia status (encoded as 0–3)  

**Target Variable**

- `target` – Heart disease diagnosis (1 = heart disease present, 0 = no heart disease)

**heart.csv**
since the data set used is very small , the actual data set can be uploaded it wont hit the free limit of streamlit . Use the file heart.csv int eh repository for testign the model


## Models Used and Evaluation Metrics


| ML Model Name        | Accuracy | AUC      | Precision | Recall  | F1      | MCC      |
|----------------------|----------|----------|-----------|---------|---------|----------|
| Logistic Regression  | 0.8098   | 0.9298   | 0.7619    | 0.9143  | 0.8312  | 0.6309   |
| Decision Tree        | 0.9854   | 0.9857   | 1.0000    | 0.9714  | 0.9855  | 0.9712   |
| kNN                  | 0.8634   | 0.9629   | 0.8738    | 0.8571  | 0.8654  | 0.7269   |
| Naive Bayes          | 0.8293   | 0.9043   | 0.8070    | 0.8762  | 0.8402  | 0.6602   |
| Random Forest        | 1.0000   | 1.0000   | 1.0000    | 1.0000  | 1.0000  | 1.0000   |
| XGBoost              | 1.0000   | 1.0000   | 1.0000    | 1.0000  | 1.0000  | 1.0000   |



### Observations on Model Performance

| ML Model Name       | Observation about model performance |
|---------------------|--------------------------------------|
| Logistic Regression | Achieves good overall performance with high AUC and recall but slightly lower precision compared to tree-based ensembles, making it a strong, simple baseline model. |

| Decision Tree       | Fits the data very closely with near-perfect accuracy and high MCC, indicating possible overfitting on this dataset despite strong test results. |

| kNN                 | Performs better than Logistic Regression in precision but slightly lower in recall, showing reasonable performance that depends strongly on scaling and the choice of k. |

| Naive Bayes         | Gives solid accuracy and recall but slightly lower AUC and MCC than kNN and Logistic Regression, reflecting its simple independence assumptions. |

| Random Forest       | Reaches perfect scores on all metrics, suggesting excellent fit but also a high chance of overfitting to this dataset due to its ensemble of many deep trees. |

| XGBoost             | Also attains perfect metrics similar to Random Forest, indicating a very powerful ensemble that may be overfitting; however it is typically strong for tabular classification tasks. |

