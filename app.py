"""Streamlit demo for the retrained PCOS model.

1. Run `python train.py` once (creates pcos_model.joblib).
2. Run `streamlit run app.py`.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
ARTIFACT = BASE / "pcos_model.joblib"
REPO_URL = "https://github.com/TayDemi/pcos-ml-detection"

st.set_page_config(page_title="PCOS prediction demo", page_icon="🩺")


@st.cache_resource
def load_artifact():
    return joblib.load(ARTIFACT)


def field(name, info):
    if info["binary"]:
        opts = info["values"]
        if opts == [0, 1]:
            return st.selectbox(name, [0, 1], format_func=lambda v: "Yes" if v == 1 else "No")
        return st.selectbox(name, opts)
    if info["whole"]:
        return st.number_input(name, int(info["min"]), int(info["max"]),
                               int(round(info["median"])), step=1)
    return st.number_input(name, float(info["min"]), float(info["max"]),
                           float(info["median"]), step=0.1)


st.title("PCOS prediction demo")
st.info(
    "This is a machine learning portfolio project, not a medical tool. "
    "It cannot diagnose PCOS. Anyone with health concerns should see a clinician."
)

if not ARTIFACT.exists():
    st.error("pcos_model.joblib not found. Run `python train.py` first.")
    st.stop()

art = load_artifact()
model, feats, info, metrics = art["model"], art["features"], art["feature_info"], art["metrics"]

with st.form("inputs"):
    values = {name: field(name, info[name]) for name in feats}
    submitted = st.form_submit_button("Predict")

if submitted:
    X = pd.DataFrame([[values[f] for f in feats]], columns=feats)
    pred = int(model.predict(X)[0])
    p = float(model.predict_proba(X)[0][list(model.classes_).index(1)])
    st.metric("Model probability of PCOS", f"{p:.0%}")
    if pred == 1:
        st.warning("For these inputs, the model predicts PCOS is likely present.")
    else:
        st.success("For these inputs, the model predicts PCOS is unlikely.")

with st.expander("How well does this model perform?"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['accuracy']:.0%}")
    c2.metric("Recall", f"{metrics['recall']:.0%}")
    c3.metric("ROC AUC", f"{metrics['roc_auc']:.2f}")
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    st.write(
        f"Measured once on {metrics['n_test']} held-out records that were not used "
        f"for training or tuning. Of these, the model caught {tp} PCOS cases, missed {fn}, "
        f"and flagged {fp} people without PCOS."
    )
    st.write("A random forest trained on a public Kaggle dataset. Features were chosen "
             "by recursive feature elimination, using cross-validation on training data only.")
    st.write("Features used: " + ", ".join(feats))
    st.markdown(f"[Code on GitHub]({REPO_URL})")
