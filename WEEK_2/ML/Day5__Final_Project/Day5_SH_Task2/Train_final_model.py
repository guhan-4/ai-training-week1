"""Train and evaluate one final churn model, once, on a protected 20% test set.

Run from the project root:
  python -m pip install -r outputs/requirements.txt
  python outputs/train_final_model.py
  python outputs/train_final_model.py --data path/to/Telco-Customer-Churn.csv

Writes final_model.joblib, final_test_predictions.csv, final_metrics.json,
and final_report.md under outputs/. The report is populated with actual run results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
TARGET = "Churn"
SEED = 42
TEST_SIZE = 0.20
TARGET_RECALL = 0.75


def fetch_or_read(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {path}")
    frame = pd.read_csv(DATA_URL)
    frame.to_csv(path, index=False)
    return frame


def prepare(frame: pd.DataFrame):
    if TARGET not in frame:
        raise ValueError(f"Dataset must have a {TARGET!r} target column.")
    if "TotalCharges" in frame:
        frame["TotalCharges"] = pd.to_numeric(
            frame["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce"
        )
    values = frame[TARGET]
    if set(values.dropna().unique()).issubset({"Yes", "No"}):
        y = values.map({"No": 0, "Yes": 1}).astype(int)
    elif set(values.dropna().unique()).issubset({0, 1}):
        y = values.astype(int)
    else:
        raise ValueError("Churn values must be Yes/No or 0/1.")
    X = frame.drop(columns=[TARGET, "customerID"], errors="ignore")
    return X, y


def pipeline_for(estimator, numeric, categorical):
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def threshold_at_recall(y, probability, target=TARGET_RECALL):
    # Highest threshold among those meeting recall requirement, to limit unnecessary outreach.
    candidates = np.unique(probability)
    qualified = [(float(t), recall_score(y, probability >= t, zero_division=0)) for t in candidates]
    valid = [(t, r) for t, r in qualified if r >= target]
    return max(valid, key=lambda pair: pair[0]) if valid else (0.0, float(recall_score(y, probability >= 0, zero_division=0)))


def write_report(path: Path, data, model_name, threshold, metrics, cm):
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    text = f"""# Customer churn prediction — final report

## Problem
Identify customers at risk of cancelling so the retention team can prioritize outreach. The operational objective is to catch at least {TARGET_RECALL:.0%} of churners, while using the highest threshold that meets that recall target on a training-only validation set.

## Data
IBM Telco Customer Churn sample: {len(data):,} customer records and {data.shape[1]-1} input columns. The `Churn` label is the target. `customerID` is excluded as an identifier; blank `TotalCharges` values are treated as missing and imputed within the model pipeline. Rows are assumed independent. A stratified 80/20 train/test split was made before model selection; the test set was evaluated only once at the end.

## Method
Compared Logistic Regression, Random Forest, and XGBoost. Used five-fold stratified CV on training data, selected the top two by CV ROC-AUC, and tuned them with `RandomizedSearchCV(scoring='roc_auc')`. Selected the final candidate using an internal training validation split. XGBoost used `scale_pos_weight` to account for class imbalance. Preprocessing (median/mode imputation, scaling and one-hot encoding) is fitted inside each pipeline fold. The final classifier was refit on all 80% training rows.

## Results
Chosen model: **{model_name}**. Validation-selected threshold: **{threshold:.4f}**. On the untouched test set: ROC-AUC **{metrics['roc_auc']:.4f}**, F1 **{metrics['f1']:.4f}**, recall **{metrics['recall']:.4f}**. Confusion matrix (actual rows No/Yes; predicted columns No/Yes): `[[{tn}, {fp}], [{fn}, {tp}]]`.

## Limitations
This is a single public benchmark snapshot; results may not generalize to another provider, period, or customer population. A random split does not test future-period drift. The recall target is a proxy for business value: this model does not include outreach cost, offer cost, customer lifetime value, or intervention success. Feature importance and model scores are predictive associations, not causal explanations.

## Recommendations
Pilot the scored list with a capped retention campaign, monitor contact capacity and offer acceptance, and compare incremental retention against a randomized control group. Review precision and campaign cost alongside recall. Revalidate on a recent time-based holdout before production use, monitor drift, and retrain when performance changes. Do not treat a risk score as a reason to deny service or price unfairly.
"""
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/Telco-Customer-Churn.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--search-iterations", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = fetch_or_read(args.data)
    X, y = prepare(raw)

    # Immediately isolate the 20% test set. Do not inspect or use it until final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    # Internal validation stays within the training partition, for selection/threshold only.
    X_dev, X_val, y_dev, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=SEED
    )
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    pos_weight = float((y_dev == 0).sum() / (y_dev == 1).sum())
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=SEED, n_jobs=1),
        "XGBoost": XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist", scale_pos_weight=pos_weight, random_state=SEED, n_jobs=1),
    }
    distributions = {
        "Logistic Regression": {"model__C": np.logspace(-2, 2, 30), "model__solver": ["liblinear", "lbfgs"]},
        "Random Forest": {"model__n_estimators": [200, 350, 500], "model__max_depth": [None, 5, 10, 16], "model__min_samples_split": [2, 5, 10], "model__min_samples_leaf": [1, 2, 4, 8], "model__max_features": ["sqrt", "log2", 0.7]},
        "XGBoost": {"model__n_estimators": [150, 250, 400, 600], "model__max_depth": [2, 3, 4, 5, 6], "model__learning_rate": [0.02, 0.04, 0.07, 0.1, 0.15], "model__subsample": [0.7, 0.85, 1.0], "model__colsample_bytree": [0.7, 0.85, 1.0], "model__min_child_weight": [1, 3, 5, 8], "model__reg_lambda": [1, 3, 7, 12]},
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    baseline_auc = {}
    for name, estimator in candidates.items():
        pipe = pipeline_for(estimator, numeric, categorical)
        baseline_auc[name] = cross_val_score(pipe, X_dev, y_dev, cv=cv, scoring="roc_auc", n_jobs=1).mean()
    selected_for_tuning = sorted(baseline_auc, key=baseline_auc.get, reverse=True)[:2]
    print("Baseline CV ROC-AUC:", {k: round(float(v), 4) for k, v in baseline_auc.items()})
    print("Tuning top two:", selected_for_tuning)

    validation_auc, thresholds, best_search_params = {}, {}, {}
    for name, estimator in candidates.items():
        pipe = pipeline_for(estimator, numeric, categorical)
        if name in selected_for_tuning:
            search = RandomizedSearchCV(
                pipe, distributions[name], n_iter=args.search_iterations, scoring="roc_auc",
                cv=cv, refit=True, random_state=SEED, n_jobs=-1, verbose=1,
            )
            search.fit(X_dev, y_dev)
            selected = search.best_estimator_
            best_search_params[name] = search.best_params_
            print(f"{name} tuned CV ROC-AUC: {search.best_score_:.4f}")
        else:
            selected = pipe.fit(X_dev, y_dev)
        p_val = selected.predict_proba(X_val)[:, 1]
        validation_auc[name] = float(roc_auc_score(y_val, p_val))
        thresholds[name], _ = threshold_at_recall(y_val.to_numpy(), p_val)

    chosen_name = max(validation_auc, key=validation_auc.get)
    print(f"Chosen on training validation ROC-AUC: {chosen_name}")
    final_pipe = pipeline_for(candidates[chosen_name], numeric, categorical)
    if chosen_name in best_search_params:
        final_pipe.set_params(**best_search_params[chosen_name])
    final_pipe.fit(X_train, y_train)
    chosen_threshold = thresholds[chosen_name]

    # First and only evaluation of the held-out test set.
    test_probability = final_pipe.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= chosen_threshold).astype(int)
    cm = confusion_matrix(y_test, test_prediction, labels=[0, 1])
    metrics = {
        "model": chosen_name,
        "threshold": float(chosen_threshold),
        "roc_auc": float(roc_auc_score(y_test, test_probability)),
        "f1": float(f1_score(y_test, test_prediction, zero_division=0)),
        "recall": float(recall_score(y_test, test_prediction, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "test_rows": int(len(y_test)),
        "threshold_target_recall": TARGET_RECALL,
    }
    print("\nFinal held-out test evaluation (once):")
    print(json.dumps(metrics, indent=2))

    bundle = {
        "model": final_pipe,
        "threshold": chosen_threshold,
        "model_name": chosen_name,
        "feature_columns": X.columns.tolist(),
        "target_column": TARGET,
    }
    joblib.dump(bundle, args.output_dir / "final_model.joblib")
    pd.DataFrame({
        "actual_churn": y_test.to_numpy(),
        "predicted_churn": test_prediction,
        "churn_probability": test_probability,
    }, index=y_test.index).to_csv(args.output_dir / "final_test_predictions.csv", index_label="row_index")
    (args.output_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(args.output_dir / "final_report.md", raw, chosen_name, chosen_threshold, metrics, cm)
    print("Saved final_model.joblib, final_test_predictions.csv, final_metrics.json, final_report.md")


if __name__ == "__main__":
    main()
