"""Task 2: feature engineering and baseline churn models.

Run from the project root:
    python outputs/task2_features_baselines.py
    python outputs/task2_features_baselines.py --data path/to/Telco-Customer-Churn.csv

Creates outputs/task2_cv_results.csv. The 20% test set is split off immediately
and is deliberately not used in the 5-fold cross-validation or feature fitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Churn"


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add domain features without learning anything from validation/test rows."""

    SERVICE_COLUMNS = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()

        tenure = pd.to_numeric(result.get("tenure"), errors="coerce")
        total = pd.to_numeric(result.get("TotalCharges"), errors="coerce")
        monthly = pd.to_numeric(result.get("MonthlyCharges"), errors="coerce")

        # Feature 1: tenure group
        result["TenureGroup"] = pd.cut(
            tenure,
            bins=[-np.inf, 12, 24, 48, np.inf],
            labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
        ).astype("object")

        # Feature 2: total spend per active month (safe for zero tenure)
        result["SpendPerTenureMonth"] = total / tenure.where(tenure > 0)

        # Feature 3: count subscribed services; yes/no phone/service flags count as 1/0.
        service_flags = []
        for column in self.SERVICE_COLUMNS:
            if column in result.columns:
                service_flags.append(result[column].astype("string").str.lower().eq("yes").astype(int))
        result["ServiceCount"] = sum(service_flags) if service_flags else 0

        # Feature 4: month-to-month contract indicator
        if "Contract" in result.columns:
            result["IsMonthToMonth"] = result["Contract"].eq("Month-to-month").astype(int)
        else:
            result["IsMonthToMonth"] = 0

        # Feature 5: monthly charges relative to the dataset's known monthly fee.
        # This ratio is available at prediction time and does not use the target.
        result["MonthlyToTotalChargeRatio"] = monthly / total.where(total > 0)

        return result


def load_data(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {path} ...")
    pd.read_csv(DATA_URL).to_csv(path, index=False)
    return pd.read_csv(path)


def make_pipeline(classifier, feature_columns: list[str]) -> Pipeline:
    """Assemble feature engineering, preprocessing, and estimator."""
    engineered_numeric = [
        "SpendPerTenureMonth", "ServiceCount", "IsMonthToMonth",
        "MonthlyToTotalChargeRatio",
    ]
    numeric_columns = [
        c for c in feature_columns if c in {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}
    ] + engineered_numeric
    categorical_columns = [c for c in feature_columns if c not in numeric_columns]
    categorical_columns += ["TenureGroup"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_columns),
        ("categorical", categorical_pipe, categorical_columns),
    ])

    return Pipeline([
        ("feature_engineering", ChurnFeatureEngineer()),
        ("preprocessing", preprocess),
        ("model", classifier),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        help="CSV path (downloads the IBM Telco sample if missing)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data)
    if TARGET not in data.columns:
        raise ValueError(f"Expected target column {TARGET!r}.")
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce"
        )

    target = data[TARGET]
    if set(target.dropna().unique()).issubset({"Yes", "No"}):
        y = target.map({"No": 0, "Yes": 1}).astype(int)
    elif set(target.dropna().unique()).issubset({0, 1}):
        y = target.astype(int)
    else:
        raise ValueError("Target must contain only Yes/No or 0/1 values.")

    X = data.drop(columns=[TARGET, "customerID"], errors="ignore")

    # Hold out test rows before CV. X_test/y_test are not accessed below.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Dataset shape: {data.shape}")
    print(f"Training rows for CV: {len(X_train)}; untouched test rows: {len(X_test)}")
    print("Engineered features: TenureGroup, SpendPerTenureMonth, ServiceCount, IsMonthToMonth, MonthlyToTotalChargeRatio")

    # Cross-validation is exclusively on the 80% training partition.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": make_scorer(f1_score, zero_division=0),
    }
    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1, min_samples_leaf=2,
        ),
    }

    result_rows = []
    for model_name, estimator in estimators.items():
        pipeline = make_pipeline(estimator, X_train.columns.tolist())
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring,
            return_train_score=False, n_jobs=1,
        )
        result_rows.append({
            "Model": model_name,
            "CV ROC-AUC": scores["test_roc_auc"].mean(),
            "CV F1": scores["test_f1"].mean(),
        })

    results = pd.DataFrame(result_rows)
    results["CV ROC-AUC"] = results["CV ROC-AUC"].round(4)
    results["CV F1"] = results["CV F1"].round(4)
    print("\n=== 5-fold cross-validation results (training partition only) ===")
    print(results.to_string(index=False))

    output_path = args.output_dir / "task2_cv_results.csv"
    results.to_csv(output_path, index=False)
    print(f"\nSaved results table to: {output_path.resolve()}")
    print("The held-out test set was not evaluated in Task 2; reserve it for the final pipeline evaluation.")


if __name__ == "__main__":
    main()
