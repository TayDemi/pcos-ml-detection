"""Streamlit demo for the PCOS random forest model.

Put this file in the repo root, next to the .pkl model and the CSV.
Run locally:  streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_FILE = "PCOS_RF_MODEL.pkl"   # saved with joblib as (model, selected_features)
DATA_FILE = "dataset-project.csv"  # only used to set sensible input ranges
REPO_URL = "https://github.com/TayDemi/pcos-ml-detection"

st.set_page_config(page_title="PCOS prediction demo", page_icon="🩺")


@st.cache_resource
def load_model():
    model, feats = joblib.load(BASE / MODEL_FILE)
    return model, list(feats)


@st.cache_data
def load_reference():
    df = pd.read_csv(BASE / DATA_FILE)
    return df.apply(pd.to_numeric, errors="coerce")


# Same bands used when the model was trained
def age_group(age):
    if age < 20:
        return "Teen"
    if age < 30:
        return "Young Adult"
    if age < 40:
        return "Adult"
    return "Older Adult"


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 24.9:
        return "Normal"
    if bmi < 29.9:
        return "Overweight"
    return "Obese"


def build_row(values, feats):
    row = dict(values)
    for f in feats:
        if f.startswith("Age_Group_"):
            row[f] = int(f == f"Age_Group_{age_group(values['Age'])}")
        elif f.startswith("BMI_Category_"):
            row[f] = int(f == f"BMI_Category_{bmi_category(values['BMI'])}")
    return pd.DataFrame([[row[f] for f in feats]], columns=feats)


def field(name, ref):
    label = name.replace("_", " ")
    col = ref[name].dropna() if name in ref else pd.Series(dtype=float)
    if col.empty:
        return st.number_input(label, value=0.0)
    if col.nunique() <= 2:
        opts = sorted(col.unique().tolist())
        if opts == [0, 1]:
            return st.selectbox(label, [0, 1], format_func=lambda v: "Yes" if v == 1 else "No")
        return st.selectbox(label, opts)
    lo, hi, mid = float(col.min()), float(col.max()), float(col.median())
    if bool((col % 1 == 0).all()):
        return st.number_input(label, int(lo), int(hi), int(round(mid)), step=1)
    return st.number_input(label, lo, hi, mid, step=0.1)


model, feats = load_model()
ref = load_reference()

st.title("PCOS prediction demo")
st.info(
    "This is a machine learning portfolio project, not a medical tool. "
    "It cannot diagnose PCOS. Anyone with health concerns should see a clinician."
)

derived = [f for f in feats if f.startswith(("Age_Group_", "BMI_Category_"))]
inputs = [f for f in feats if f not in derived]
if any(f.startswith("Age_Group_") for f in derived) and "Age" not in inputs:
    inputs.append("Age")
if any(f.startswith("BMI_Category_") for f in derived) and "BMI" not in inputs:
    inputs.append("BMI")

with st.form("inputs"):
    values = {name: field(name, ref) for name in inputs}
    submitted = st.form_submit_button("Predict")

if submitted:
    X = build_row(values, feats)
    pred = int(model.predict(X)[0])
    if hasattr(model, "predict_proba") and 1 in list(model.classes_):
        p = float(model.predict_proba(X)[0][list(model.classes_).index(1)])
        st.metric("Model probability of PCOS", f"{p:.0%}")
    if pred == 1:
        st.warning("For these inputs, the model predicts PCOS is likely present.")
    else:
        st.success("For these inputs, the model predicts PCOS is unlikely.")

with st.expander("About this model"):
    st.write(
        "A random forest trained on a public Kaggle dataset of clinical and "
        "hormonal attributes, with features chosen by recursive feature "
        "elimination. See the repository for the code and evaluation."
    )
    st.write("Features used: " + ", ".join(feats))
    st.markdown(f"[Code on GitHub]({REPO_URL})")
