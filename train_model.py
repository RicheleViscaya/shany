# train_model.py
# Student Performance Prediction System - data retrieval, training, evaluation
import io
import json
import urllib.request
import zipfile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------------------------
# ACTIVITY A - Retrieve and inspect the dataset
# ---------------------------------------------------------------------------
MATH_ROWS = 395  # the Mathematics file (student-mat.csv) has 395 records
ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def read_math_from_zip(raw_bytes):
    """Search a (possibly nested) UCI zip for student-mat.csv."""
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
        for name in z.namelist():
            if name.endswith("student-mat.csv"):
                return pd.read_csv(z.open(name), sep=";")
        for name in z.namelist():
            if name.endswith(".zip"):
                found = read_math_from_zip(z.read(name))
                if found is not None:
                    return found
    return None


# Fetch UCI Student Performance dataset (ID 320)
student = fetch_ucirepo(id=320)
df = pd.concat([student.data.features, student.data.targets], axis=1)

# The UCI package may return the Portuguese file; make sure we use Mathematics.
if len(df) != MATH_ROWS or not {"G1", "G2", "G3"}.issubset(df.columns):
    print(f"Note: fetched {len(df)} rows. Trying the Mathematics file directly...")
    try:
        with urllib.request.urlopen(ZIP_URL, timeout=60) as resp:
            math_df = read_math_from_zip(resp.read())
        if math_df is not None:
            df = math_df
            print("Loaded student-mat.csv (Mathematics records).")
        else:
            print("WARNING: student-mat.csv not found; using the fetched data.")
    except Exception as exc:  # network problems etc.
        print("WARNING: could not download Mathematics file:", exc)

df.to_csv("student_data.csv", index=False)
print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())
assert {"G1", "G2", "G3"}.issubset(df.columns), "G1, G2 and G3 must be present"

# ---------------------------------------------------------------------------
# ACTIVITY B - Preprocess, train, and evaluate
# ---------------------------------------------------------------------------
# Target: 1 = at risk (G3 < 10); 0 = not at risk
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )),
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = round(accuracy_score(y_test, predictions), 3)
names = ["Not at risk", "At risk"]
print("Accuracy:", accuracy)
print(classification_report(y_test, predictions, target_names=names, zero_division=0))

# Save the metrics so they can be pasted into the submission
report = classification_report(
    y_test, predictions, target_names=names, zero_division=0, output_dict=True
)
with open("evaluation_results.json", "w") as f:
    json.dump({"accuracy": accuracy, "report": report}, f, indent=2)
with open("evaluation_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(classification_report(
        y_test, predictions, target_names=names, zero_division=0
    ))

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl and evaluation_results.txt")
