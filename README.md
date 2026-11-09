# Student Performance Prediction System

A classroom prototype that flags students who may be at risk of failing the final Mathematics grade, built for the Data Mining laboratory (Python, GitHub, Streamlit, SAD).

## Problem
Academic risk is often noticed only after poor results. This prototype explores early identification of students who may need additional academic support using available student indicators.

## Dataset
[UCI Machine Learning Repository - Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) (Mathematics records, `student-mat.csv`, 395 rows). It contains demographic, social, and academic attributes, including the period grades G1, G2 and the final grade G3.

## Prediction target
- `G3 < 10` -> **AT RISK** (1)
- `G3 >= 10` -> **NOT AT RISK** (0)

`G3` is used only to build the target and is **never** an input feature (to avoid target leakage).

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest), joblib, ucimlrepo, Streamlit, Git/GitHub.

## Installation and run
```bash
git clone https://github.com/YOUR-USERNAME/student-performance-prediction.git
cd student-performance-prediction
pip install -r requirements.txt

python train_model.py      # downloads data, trains model, creates student_data.csv,
                           # student_model.pkl and evaluation_results.txt
streamlit run app.py       # opens http://localhost:8501
```
In the app, download the sample input CSV, upload it, review the predictions and download the results.

## Model evaluation
See `evaluation_results.txt` after running `train_model.py` (accuracy, precision, recall, F1-score). Accuracy alone should not be used to judge the model; recall for the "At risk" class is especially important.

## Limitations and responsible use
- Trained on one Portuguese secondary-school dataset; it may not generalize to other institutions or populations.
- Small dataset (about 395 records), so metrics can vary between splits.
- G1 and G2 are strong predictors, so the model is less useful for very early prediction.
- Predictions are **not diagnoses** and must not be the sole basis for real student decisions.
