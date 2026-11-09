# app.py  (self-contained: loads data, trains the model if needed, runs the UI)
import io
import os
import urllib.request
import zipfile

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Student Predict", page_icon="🎓", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "student_model.pkl")
ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def _find_math(raw_bytes):
    """Look for student-mat.csv inside the (possibly nested) UCI zip."""
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
        for name in z.namelist():
            if name.endswith("student-mat.csv"):
                return pd.read_csv(z.open(name), sep=";")
        for name in z.namelist():
            if name.endswith(".zip"):
                found = _find_math(z.read(name))
                if found is not None:
                    return found
    return None


def load_data():
    """Use student_data.csv if present; otherwise download it from UCI."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    df = None
    try:
        with urllib.request.urlopen(ZIP_URL, timeout=60) as resp:
            df = _find_math(resp.read())
    except Exception:
        df = None
    if df is None:  # fallback: ucimlrepo package
        from ucimlrepo import fetch_ucirepo

        student = fetch_ucirepo(id=320)
        df = pd.concat([student.data.features, student.data.targets], axis=1)
    df.to_csv(DATA_PATH, index=False)
    return df


def train_model(df):
    """Same pipeline as train_model.py (G3 is the target only, never a feature)."""
    df = df.drop_duplicates()
    y = (df["G3"] < 10).astype(int)
    X = df.drop(columns=["G3"])
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )),
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model


@st.cache_resource(show_spinner="Preparing model (first run only, please wait)...")
def prepare():
    data = load_data()
    expected = [c for c in data.columns if c != "G3"]
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            if sorted(model.feature_names_in_) != sorted(expected):
                model = None
        except Exception:
            model = None
    if model is None:
        model = train_model(data)
    return model, data


try:
    model, data = prepare()
except Exception as exc:
    st.error("Could not prepare the model. Details below - please send a screenshot.")
    st.exception(exc)
    st.stop()

st.title("🎓 Student Performance Prediction")
st.caption("Classroom prototype for academic risk exploration  |  app version v3")
st.warning(
    "Predictions are not diagnoses and must not be used as the sole basis "
    "for real student decisions."
)

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average final grade", round(data["G3"].mean(), 2))
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict uploaded student records")
st.write("Upload a CSV containing the same predictor columns as the training data.")

template = data.drop(columns=["G3"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_prediction_input.csv",
    "text/csv",
)

uploaded = st.file_uploader("Choose student CSV", type=["csv"])

if uploaded is not None:
    new_data = pd.read_csv(uploaded)
    required_columns = list(model.feature_names_in_)
    missing = [c for c in required_columns if c not in new_data.columns]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        predictions = model.predict(new_data[required_columns])
        results = new_data.copy()
        results["Prediction"] = [
            "AT RISK" if value == 1 else "NOT AT RISK" for value in predictions
        ]

        st.subheader("Prediction results")
        st.dataframe(results, use_container_width=True)
        st.metric("Predicted at-risk records", int((predictions == 1).sum()))
        st.download_button(
            "Download prediction results",
            results.to_csv(index=False),
            "student_predictions.csv",
            "text/csv",
        )
