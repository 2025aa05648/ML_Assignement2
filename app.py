# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import joblib
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


st.title("Indian Bank Loan Prediction System")

#Upload Dataset

uploaded_file = st.file_uploader("Upload Test CSV File", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

   # X = df.drop("target", axis=1)
    X = df.copy()
   # y = df["target"]
   # if "Loan_Status" in df.columns:
      #  df = df.drop("Loan_Status", axis=1)
  #1 if "Loan_Status" not in df.columns:
       #2 st.error("Please upload test dataset including Loan_Status column for evaluation.")
    #3else:
       #4 y_true = df["Loan_Status"]
    # Check if target exists
    if "Loan_Status" in df.columns:
        y_true = df["Loan_Status"]
    
        # Convert target to numeric if needed
        y_true = y_true.map({"Y": 1, "N": 0})
    
        # Drop target from features
        df = df.drop("Loan_Status", axis=1)
    
    else:
        y_true = None
    
    # Model selection
    model_choice = st.selectbox("Choose a model",
                                ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"])
    #Model filenames
    model_files = {
    "Logistic Regression": "model/lr.pkl",
    "Decision Tree": "model/dt.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/nb.pkl",
    "Random Forest": "model/rf.pkl",
    "XGBoost": "model/xgb.pkl"
}

    model_path = model_files[model_choice]

    model = joblib.load(model_path)
    # Load model
   # model = pickle.load(open(f"model/{model_choice.replace(' ', '_').lower()}.pkl", "rb"))
    #Preprocessing Data
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # Encode categorical columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    # Scale
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    
    # Predictions
   #y_pred = model.predict(X)
    y_pred = model.predict(df_scaled)

    # Metrics
    st.subheader("Evaluation Metrics")
    st.text(classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
