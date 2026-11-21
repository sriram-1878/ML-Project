import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction")
st.write("A machine-learning application that estimates whether a customer may leave a service.")

@st.cache_data
def load_sample():
    return pd.read_csv("data/customer_churn_sample.csv")

@st.cache_resource
def train_model(data):
    target = "churn"
    X = data.drop(columns=[target])
    y = data[target].astype(int)
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=180, random_state=42, class_weight="balanced"
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, X.columns.tolist(), accuracy, y_test, predictions

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload your customer CSV (optional)", type=["csv"])
    st.caption("The included sample dataset is used if no file is uploaded.")
    st.markdown("**Required target column:** `churn` (0 = No, 1 = Yes)")

try:
    data = pd.read_csv(uploaded) if uploaded else load_sample()
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if "churn" not in data.columns:
    st.error("Your CSV must contain a `churn` column with 0/1 values.")
    st.stop()

data = data.dropna(subset=["churn"]).copy()
data["churn"] = pd.to_numeric(data["churn"], errors="coerce")
data = data.dropna(subset=["churn"])
data["churn"] = data["churn"].astype(int)
data = data[data["churn"].isin([0, 1])]
if len(data) < 20 or data["churn"].nunique() < 2:
    st.error("Please provide at least 20 usable rows and both churn classes (0 and 1).")
    st.stop()

st.subheader("Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Customers", f"{len(data):,}")
c2.metric("Features", len(data.columns) - 1)
c3.metric("Churn rate", f"{data['churn'].mean():.1%}")
with st.expander("Preview customer data"):
    st.dataframe(data.head(20), use_container_width=True)

model, feature_columns, accuracy, y_test, predictions = train_model(data)
st.subheader("Model performance")
st.metric("Test accuracy", f"{accuracy:.1%}")
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=["Stayed", "Churned"], ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig)
plt.close(fig)

left, right = st.columns(2)
with left:
    st.markdown("**Customer churn distribution**")
    counts = data["churn"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots()
    ax.bar(["Stayed", "Churned"], counts.values)
    ax.set_ylabel("Number of customers")
    st.pyplot(fig)
    plt.close(fig)
with right:
    if "contract" in data.columns:
        st.markdown("**Churn rate by contract**")
        rates = data.groupby("contract")["churn"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots()
        rates.plot(kind="bar", ax=ax)
        ax.set_ylabel("Churn rate")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig)
        plt.close(fig)

st.subheader("Predict churn for a customer")
st.write("Enter values below. The fields are based on the included sample dataset.")
sample = data.drop(columns=["churn"])
user_values = {}
cols = st.columns(2)
for i, col in enumerate(feature_columns):
    with cols[i % 2]:
        if pd.api.types.is_numeric_dtype(sample[col]):
            default = float(pd.to_numeric(sample[col], errors="coerce").median())
            user_values[col] = st.number_input(col.replace("_", " ").title(), value=default)
        else:
            options = sorted(sample[col].dropna().astype(str).unique().tolist())
            if not options:
                options = ["Unknown"]
            user_values[col] = st.selectbox(col.replace("_", " ").title(), options)

if st.button("Predict customer churn", type="primary"):
    row = pd.DataFrame([user_values], columns=feature_columns)
    probability = float(model.predict_proba(row)[0][1])
    prediction = int(probability >= 0.5)
    if prediction:
        st.warning(f"Prediction: Customer may churn (estimated probability: {probability:.1%}).")
        st.write("Possible action: review the customer's experience and consider suitable support or retention options.")
    else:
        st.success(f"Prediction: Customer is likely to stay (estimated churn probability: {probability:.1%}).")
    st.progress(probability)

st.caption("Educational prototype. Predictions are estimates, not guarantees. Validate with representative real-world data before business use.")
