# Student Performance Prediction System

## Problem
Academic risk is often noticed only after poor results. This classroom prototype explores whether historical student indicators can flag learners who may need additional academic support.

## Dataset
UCI Machine Learning Repository – Student Performance (ID 320): https://archive.ics.uci.edu/dataset/320/student+performance

## Prediction target
`G3 < 10` → **AT RISK**; `G3 >= 10` → **NOT AT RISK**. `G3` is the target only and is never used as an input feature.

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest), joblib, Streamlit, GitHub.

## Installation and run
```bash
pip install -r requirements.txt
python train_model.py        # downloads data, trains model, saves student_model.pkl
streamlit run app.py         # opens http://localhost:8501
```
In the app, download the sample input CSV, then upload it to see predictions.

## Model results
Fill in after running `train_model.py`: accuracy, precision, recall, F1-score (At risk class).

## Limitations
- Trained on one Portuguese secondary-school dataset; may not generalize to other institutions or populations.
- Predictions are not diagnoses and must not be the sole basis for real student decisions.
- G1 and G2 are strong predictors, so the model flags risk late in the term.
