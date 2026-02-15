import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Loan Prediction App", layout="wide")

st.title("🏦 Indian Bank Loan Prediction")
st.write("Upload test dataset (CSV) to predict Loan Status.")

# ------------------------------
# Model Selection
# ------------------------------

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression",
     "Decision Tree",
     "KNN",
     "Naive Bayes",
     "Random Forest",
     "XGBoost"]
)

model_files = {
    "Logistic Regression": "model/lr.pkl",
    "Decision Tree": "model/dt.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/nb.pkl",
    "Random Forest": "model/rf.pkl",
    "XGBoost": "model/xgb.pkl"
}

model = joblib.load(model_files[model_choice])

# ------------------------------
# File Upload
# ------------------------------

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # ------------------------------
    # Extract Target if Available
    # ------------------------------

    if "Loan_Status" in df.columns:
        y_true = df["Loan_Status"].map({"Y": 1, "N": 0})
        df = df.drop("Loan_Status", axis=1)
    else:
        y_true = None

    # Remove Loan_ID if present
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # ------------------------------
    # Handle Missing Values
    # ------------------------------

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # ------------------------------
    # Encode Categorical Variables
    # ------------------------------

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    # ------------------------------
    # Scale Features
    # ------------------------------

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # ------------------------------
    # Prediction
    # ------------------------------

    y_pred = model.predict(df_scaled)

    # Convert predictions back to labels
    prediction_labels = ["Y" if pred == 1 else "N" for pred in y_pred]

    result_df = df.copy()
    result_df["Predicted_Loan_Status"] = prediction_labels

    st.subheader("📌 Prediction Results")
    st.dataframe(result_df)

    # ------------------------------
    # Evaluation Metrics (If True Labels Exist)
    # ------------------------------

    if y_true is not None:

        st.subheader("📊 Classification Report")
        st.text(classification_report(y_true, y_pred))

        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        st.write(cm)

    else:
        st.info("Loan_Status column not found. Showing predictions only.")
