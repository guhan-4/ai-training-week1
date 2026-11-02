"""Task 3: XGBoost, tuning, model comparison, thresholding and explanations.

Install: python -m pip install pandas numpy matplotlib scikit-learn xgboost
Run:     python outputs/task3_xgb_tuning_comparison.py
         python outputs/task3_xgb_tuning_comparison.py --data path/to/telco.csv

Outputs: comparison_table.csv, roc_comparison.png, best_model_feature_importance.png
under outputs/ (or --output-dir). The outer 20% test set is reserved immediately
and used only for final metrics and the final ROC plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

TARGET = "Churn"

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
SEED = 42
TARGET_RECALL = 0.75  # Business objective: detect at least 75% of churners.


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Telco churn dataset to {path}")
        data = pd.read_csv(DATA_URL)
        data.to_csv(path, index=False)
        return data
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame):
    if TARGET not in df:
        raise ValueError(f"Expected target column {TARGET!r}.")
    if "TotalCharges" in df:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce"
        )
    target = df[TARGET]
    if set(target.dropna().unique()).issubset({"Yes", "No"}):
        y = target.map({"No": 0, "Yes": 1}).astype(int)
    elif set(target.dropna().unique()).issubset({0, 1}):
        y = target.astype(int)
    else:
        raise ValueError("Churn target must contain Yes/No or 0/1 values.")
    X = df.drop(columns=[TARGET, "customerID"], errors="ignore")
    return X, y


def make_pipeline(model, numeric_features, categorical_features):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def threshold_for_recall(y_true, probabilities, target_recall=TARGET_RECALL):
    """Choose the highest threshold that achieves the requested validation recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    eligible = np.flatnonzero(recall[:-1] >= target_recall)
    if len(eligible) == 0:
        return 0.0, float(recall.max())
    # Highest qualifying threshold reduces false-positive outreach while meeting recall goal.
    idx = eligible[np.argmax(thresholds[eligible])]
    return float(thresholds[idx]), float(recall[idx])


def explain_feature(name: str) -> str:
    """Translate encoded feature names into a concise business interpretation."""
    raw = name.split("__", 1)[-1]
    rules = [
        ("Contract_Month-to-month", "Month-to-month customers can leave without a long commitment; retention offers may help convert them to longer contracts."),
        ("Contract_Two year", "Two-year contract customers are generally more committed; a high importance may reflect lower churn risk."),
        ("Contract_One year", "One-year contract customers have a defined renewal point; timely renewal outreach may matter."),
        ("tenure", "Customer tenure captures relationship length; newer customers often need stronger onboarding and early support."),
        ("MonthlyCharges", "Higher monthly bills can increase price sensitivity; review plan fit and value communication."),
        ("TotalCharges", "Cumulative spend is a proxy for relationship history and customer value."),
        ("PaymentMethod_Electronic check", "Electronic-check customers may have less payment automation; offer convenient recurring payment options."),
        ("InternetService_Fiber optic", "Fiber customers may have higher service expectations; reliability and support experience matter."),
        ("TechSupport_Yes", "Technical-support enrollment may indicate service needs or a stronger support relationship; interpret with churn direction."),
        ("OnlineSecurity_Yes", "Security add-on uptake reflects product mix and customer engagement; targeted bundles may improve perceived value."),
        ("PaperlessBilling_Yes", "Paperless billing is a digital-engagement signal and may correlate with customer segment or payment behavior."),
        ("SeniorCitizen", "Age segment is a proxy for differing needs; tailor service and communications carefully."),
    ]
    for token, meaning in rules:
        if token in raw:
            return meaning
    if "_Yes" in raw:
        return f"Customers with {raw.replace('_Yes', '')} enabled differ in product adoption; consider whether a relevant bundle or support intervention is useful."
    if "_No" in raw:
        return f"Customers without {raw.replace('_No', '')} may have an unmet need or a different service profile; test a relevant offer."
    return "This feature distinguishes a customer segment; use it to guide investigation and tailored retention actions, not as a causal explanation."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/Telco-Customer-Churn.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--search-iterations", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data)
    X, y = prepare_data(data)

    # Create the untouched final test set immediately; no subsequent selection uses it.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    # Internal validation is drawn only from the 80% training partition. It chooses
    # business thresholds and the best model for interpretation.
    X_dev, X_val, y_dev, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=SEED
    )

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    positive_weight = (y_dev == 0).sum() / (y_dev == 1).sum()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", min_samples_leaf=2,
            random_state=SEED, n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            scale_pos_weight=positive_weight, random_state=SEED, n_jobs=1,
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("Finding the best two baselines by 5-fold training CV ROC-AUC ...")
    baseline_auc = {}
    for name, estimator in models.items():
        pipe = make_pipeline(estimator, numeric_features, categorical_features)
        fold_scores = cross_val_score(pipe, X_dev, y_dev, cv=cv, scoring="roc_auc", n_jobs=1)
        baseline_auc[name] = float(fold_scores.mean())
        print(f"  {name}: {fold_scores.mean():.4f} (+/- {fold_scores.std():.4f})")

    top_two = sorted(baseline_auc, key=baseline_auc.get, reverse=True)[:2]
    print("\nTuning these two with RandomizedSearchCV/scoring='roc_auc':", ", ".join(top_two))
    spaces = {
        "Logistic Regression": {
            "model__C": np.logspace(-2, 2, 30),
            "model__solver": ["liblinear", "lbfgs"],
        },
        "Random Forest": {
            "model__n_estimators": [200, 350, 500, 700],
            "model__max_depth": [None, 5, 8, 12, 18],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.7],
        },
        "XGBoost": {
            "model__n_estimators": [150, 250, 400, 600],
            "model__max_depth": [2, 3, 4, 5, 6],
            "model__learning_rate": [0.02, 0.04, 0.07, 0.1, 0.15],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__min_child_weight": [1, 3, 5, 8],
            "model__reg_lambda": [1, 3, 7, 12],
        },
    }

    # Keep fixed and tuned estimators separate; all fitted only on development rows.
    dev_models = {}
    tuned_params = {}
    for name, estimator in models.items():
        pipe = make_pipeline(estimator, numeric_features, categorical_features)
        if name in top_two:
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=spaces[name],
                n_iter=args.search_iterations,
                scoring="roc_auc",
                cv=cv,
                refit=True,
                random_state=SEED,
                n_jobs=-1,
                verbose=1,
            )
            search.fit(X_dev, y_dev)
            dev_models[name] = search.best_estimator_
            tuned_params[name] = search.best_params_
            print(f"{name} best CV ROC-AUC: {search.best_score_:.4f}")
        else:
            pipe.fit(X_dev, y_dev)
            dev_models[name] = pipe

    # Use internal validation only for threshold selection and model choice.
    validation_info = {}
    validation_auc = {}
    for name, estimator in dev_models.items():
        prob = estimator.predict_proba(X_val)[:, 1]
        threshold, achieved_recall = threshold_for_recall(y_val, prob)
        validation_info[name] = {"threshold": threshold, "recall": achieved_recall}
        validation_auc[name] = roc_auc_score(y_val, prob)
        print(f"{name}: validation AUC={validation_auc[name]:.4f}, threshold={threshold:.4f}, recall={achieved_recall:.3f}")

    # Select the best model using validation ROC-AUC, then refit every candidate on
    # all outer training rows. Test set is touched only below for final evaluation.
    best_name = max(validation_auc, key=validation_auc.get)
    print(f"\nBest validation model for feature interpretation: {best_name}")
    final_models = {}
    for name in models:
        final_estimator = make_pipeline(models[name], numeric_features, categorical_features)
        if name in tuned_params:
            final_estimator.set_params(**tuned_params[name])
        final_estimator.fit(X_train, y_train)
        final_models[name] = final_estimator

    comparison = []
    test_probabilities = {}
    for name, estimator in final_models.items():
        probabilities = estimator.predict_proba(X_test)[:, 1]
        test_probabilities[name] = probabilities
        threshold = validation_info[name]["threshold"]
        prediction = (probabilities >= threshold).astype(int)
        comparison.append({
            "Model": name + (" (tuned)" if name in tuned_params else " (baseline)"),
            "Test ROC-AUC": roc_auc_score(y_test, probabilities),
            "Test F1": f1_score(y_test, prediction, zero_division=0),
            "Threshold": threshold,
            "Test Recall": recall_score(y_test, prediction, zero_division=0),
        })

    comparison_df = pd.DataFrame(comparison).sort_values("Test ROC-AUC", ascending=False)
    comparison_df.to_csv(args.output_dir / "comparison_table.csv", index=False)
    print("\n=== Final comparison on untouched test set ===")
    print(comparison_df.round(4).to_string(index=False))

    # ROC chart: probability ranking; threshold-free comparison.
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, probabilities in test_probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc = roc_auc_score(y_test, probabilities)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC Curves — final 20% test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "roc_comparison.png", dpi=180)
    plt.close(fig)

    # Importances for best validation model. Tree importance for RF/XGB; absolute
    # standardized coefficient magnitude for Logistic Regression.
    best_pipe = final_models[best_name]
    fitted_preprocessor = best_pipe.named_steps["preprocess"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    fitted_model = best_pipe.named_steps["model"]
    if hasattr(fitted_model, "feature_importances_"):
        importance = fitted_model.feature_importances_
        importance_label = "Model feature importance"
    else:
        importance = np.abs(fitted_model.coef_[0])
        importance_label = "Absolute standardized coefficient (importance proxy)"
    top_idx = np.argsort(importance)[-15:][::-1]
    top_names = feature_names[top_idx]
    top_values = importance[top_idx]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh([n.replace("numeric__", "").replace("categorical__", "") for n in top_names][::-1], top_values[::-1], color="#3977a8")
    ax.set_xlabel(importance_label)
    ax.set_title(f"Top feature importance — {best_name}")
    fig.tight_layout()
    fig.savefig(args.output_dir / "best_model_feature_importance.png", dpi=180)
    plt.close(fig)

    print("\nTop 5 features and business interpretation (importance is associative, not causal):")
    explanation_lines = [
        f"Best validation model: {best_name}",
        "Feature importance indicates predictive association, not a causal driver of churn.",
        "",
    ]
    for name, value in zip(top_names[:5], top_values[:5]):
        clean_name = name.replace("numeric__", "").replace("categorical__", "")
        explanation = explain_feature(clean_name)
        print(f"\n- {clean_name} ({value:.4f}): {explanation}")
        explanation_lines.append(f"- {clean_name} ({value:.4f}): {explanation}")
    (args.output_dir / "top5_business_interpretation.txt").write_text(
        "\n".join(explanation_lines), encoding="utf-8"
    )

    print(f"\nBusiness threshold: validation-selected thresholds target >= {TARGET_RECALL:.0%} churner recall.")
    print("If the threshold cannot achieve the target on validation, the script uses 0.0; review the resulting test recall and contact capacity.")
    print("Saved table and charts to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
