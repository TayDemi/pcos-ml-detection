"""Retrain the PCOS model without test-set leakage and save one artifact for the app.

Usage:  python train.py        (expects PCOS_data.csv next to this file)
Output: pcos_model.joblib      (model, selected features, input ranges, test metrics)

What changed compared with MODEL2_COMPLETED.py:
  * Cleaning keeps decimal points. Only a trailing "." (e.g. "1.99.") is stripped,
    and anything that is still not a number becomes missing.
  * Missing values are filled with the TRAINING median, inside the pipeline.
  * The number of features (RFE) and the Random Forest settings are chosen with
    cross-validation on the training set only. The test set is used once, at the end.
  * The imputer and model are saved together, so the app needs no extra scaling steps.
"""
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).parent
DATA_FILE = BASE / "PCOS_data.csv"
OUT_FILE = BASE / "pcos_model.joblib"
TARGET = "PCOS (Y/N)"
ID_COLUMNS = ["Sl. No", "Patient File No."]
SEED = 42


def load_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    y = pd.to_numeric(df.pop(TARGET), errors="coerce")
    keep = y.notna()
    df, y = df[keep], y[keep].astype(int)
    X = df.apply(lambda s: pd.to_numeric(s.astype(str).str.strip().str.rstrip("."), errors="coerce"))
    X = X.dropna(axis=1, how="all")
    return X, y


def choose_features(X_train, y_train, cv):
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rfe", RFE(RandomForestClassifier(n_estimators=100, random_state=SEED), step=2)),
    ])
    sizes = [n for n in (5, 8, 10, 15, 20) if n <= X_train.shape[1]]
    search = GridSearchCV(pipe, {"rfe__n_features_to_select": sizes},
                          cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    mask = search.best_estimator_.named_steps["rfe"].support_
    return list(X_train.columns[mask]), search.best_params_["rfe__n_features_to_select"]


def main():
    X, y = load_clean(DATA_FILE)
    print(f"Rows: {len(X)}, columns: {X.shape[1]}, PCOS rate: {y.mean():.0%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    features, n_feats = choose_features(X_train, y_train, cv)
    print(f"Selected {n_feats} features by cross-validation: {features}")

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(random_state=SEED, class_weight="balanced")),
    ])
    grid = {"model__n_estimators": [100, 300], "model__max_depth": [None, 10, 20]}
    search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train[features], y_train)
    model = search.best_estimator_

    pred = model.predict(X_test[features])
    proba = model.predict_proba(X_test[features])[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "cv_roc_auc_train": float(search.best_score_),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "best_params": search.best_params_,
    }

    info = {}
    for c in features:
        col = X[c].dropna()
        binary = bool(col.nunique() <= 2)
        info[c] = {
            "binary": binary,
            "values": sorted(col.unique().tolist()) if binary else None,
            "min": float(col.min()), "max": float(col.max()), "median": float(col.median()),
            "whole": bool((col % 1 == 0).all()),
        }

    joblib.dump({"model": model, "features": features, "feature_info": info,
                 "metrics": metrics, "sklearn_version": sklearn.__version__}, OUT_FILE)

    print("\nHeld-out test results (use these numbers in your README):")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k}: {metrics[k]:.3f}")
    print("  confusion matrix [[TN, FP], [FN, TP]]:", metrics["confusion_matrix"])
    print(f"\nSaved {OUT_FILE.name} (scikit-learn {sklearn.__version__})")


if __name__ == "__main__":
    main()
