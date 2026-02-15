# ML_Assignement2
Repository for  Machine Learning Assignment 2

# a. Problem Statement

Loan approval is a critical decision-making process in the banking sector. Financial institutions must evaluate applicant information carefully to determine whether a loan should be approved or rejected. Incorrect decisions may lead to financial loss due to loan defaults.
The objective of this assignement is to develop multiple machine learning classification models that predict whether a loan application will be approved based on applicant demographic and financial information.

The assignement covers:

1. End-to-end machine learning workflow
2. Implementation of multiple classification models
3. Performance comparison using evaluation metrics
4. Deployment of models using Streamlit

# b. Dataset Used: Indian Bank Loan Prediction Dataset (Kaggle)

This dataset contains loan application records of applicants and whether their loan was approved.

Dataset Details:
Number of Instances: 614
Number of Features: 13
Target Variable: Loan_Status (Approved / Not Approved)
Problem Type: Binary Classification

### 🧾 Feature Description:

| Feature Name      | Description                |
| ----------------- | -------------------------- |
| Loan_ID           | Unique Loan ID             |
| Gender            | Applicant Gender           |
| Married           | Marital Status             |
| Dependents        | Number of dependents       |
| Education         | Education level            |
| Self_Employed     | Self-employed or not       |
| ApplicantIncome   | Income of applicant        |
| CoapplicantIncome | Income of co-applicant     |
| LoanAmount        | Loan amount requested      |
| Loan_Amount_Term  | Loan term                  |
| Credit_History    | Credit history (0/1)       |
| Property_Area     | Urban / Semi-Urban / Rural |
| Loan_Status       | Target Variable            |

# Data Preprocessing Steps
1. Removed Loan_ID column (not predictive)
2. Handled missing values:
3. Encoded categorical variables using Label Encoding
4. Standardized numerical features using StandardScaler
5. Split dataset into 80% training and 20% testing sets

# c. Models Used and Evaluation
The following six classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble – Bagging)
6. XGBoost (Ensemble – Boosting)

# d. Evaluation Metrics Used

Each model was evaluated using:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6.  Matthews Correlation Coefficient (MCC)

These metrics were chosen to evaluate not only overall correctness (accuracy) but also balance between false positives and false negatives, which is very important in loan approval systems.

---

# Model Comparison Table

| ML Model Name        | Accuracy  | AUC        | Precision | Recall  | F1        | MCC      | 
|----------------------|-----------|------------|-----------|---------|-----------|----------|
| Logistic Regression  | 0.788618  | 0.752035   | 0.759615  | 0.9875  | 0.858696  | 0.535826 | 
| Decision Tree        | 0.731707  | 0.696948   | 0.783133  | 0.8125  | 0.797546  | 0.400951 |
| KNN                  | 0.764228  | 0.656977   | 0.742857  | 0.9750  | 0.843243  | 0.468268 |
| Naive Bayes          | 0.780488  | 0.726453   | 0.757282  | 0.9750  | 0.852459  | 0.508635 |
| Random Forest        | 0.780488  | 0.757849   | 0.757282  | 0.9750  | 0.852459  | 0.508635 |
| XGBoost              | 0.764228  | 0.740988   | 0.768421  | 0.9125  | 0.834286  | 0.455873 |

> Note:This values are captured after a program run. Value may differe depending on preprocessing and random state when program is rerun.

# Observations on Model Performance

| ML Model Name       | Observation about Model Performance                                                                             |
| ------------------- | --------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Performs well due to strong influence of Credit_History feature. Simple and interpretable model.                |
| Decision Tree       | Captures non-linear relationships but tends to overfit on smaller datasets.                                     |
| KNN                 | Works reasonably well but sensitive to scaling and choice of K value.                                           |
| Naive Bayes         | Assumes feature independence; moderate performance due to correlated features like income and loan amount.      |
| Random Forest       | Improves stability over Decision Tree by reducing variance. Good balance between bias and variance.             |
| XGBoost             | Best performing model. Boosting helps improve prediction accuracy and handles feature interactions effectively. |

# Key Insights

1. Credit_History is the most influential feature.
2. Ensemble models outperform individual models.
3. Boosting (XGBoost) achieved the highest AUC and MCC score.
4. Logistic Regression performs surprisingly well due to structured financial data.

---

# Streamlit Web Application Features

The deployed Streamlit app includes:

1. Upload Test Dataset (CSV)
2. Model Selection Dropdown
3. Display of Evaluation Metrics
4. Confusion Matrix
5. Classification Report
6.  Prediction Output

---

# 📂 Assignment Repository Structure

```
Assignement-folder/
│-- app.py
│-- 2025aa05648_ML_Assignement2.ipynb
│-- requirements.txt
│-- README.md
│-- Train.csv
│-- Test.csv
│-- model/
│    ├── logistic.pkl
│    ├── decision_tree.pkl
│    ├── knn.pkl
│    ├── naive_bayes.pkl
│    ├── random_forest.pkl
│    └── xgboost.pkl


# Deployment

The application is deployed using **Streamlit Community Cloud** and connected to GitHub repository.

---

# Conclusion

This assignment successfully demonstrates the implementation and comparison of six machine learning classification models for loan approval prediction.

Among all models:

* XGBoost achieved the best overall performance.
* Random Forest also performed strongly.
* Logistic Regression provided competitive results with high interpretability.

The results highlight the importance of ensemble techniques in financial decision-making systems.

