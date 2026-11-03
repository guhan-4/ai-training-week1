from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


# Reads sample_loan_data.csv from the same folder as this script.
DATA_PATH = Path(__file__).with_name("sample_loan_data.csv")
TARGET_COLUMN = "default"
DATE_COLUMN = "application_date"
ORDINAL_COLUMN = "education_level"

# Explicit order for ordinal encoding.
EDUCATION_ORDER = [["high_school", "bachelor", "master", "doctoral"]]
RANDOM_STATE = 42


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create features while keeping preprocessing inside the pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Date decomposition: extract year, month, and day.
        date = pd.to_datetime(X[DATE_COLUMN], errors="coerce")
        X["application_year"] = date.dt.year
        X["application_month"] = date.dt.month
        X["application_day"] = date.dt.day

        # Interaction feature: income compared with requested loan amount.
        income = pd.to_numeric(X["annual_income"], errors="coerce")
        loan = pd.to_numeric(X["loan_amount"], errors="coerce")
        X["income_to_loan_ratio"] = income / loan.replace(0, np.nan)

        # Log transform for the usually skewed loan amount.
        X["log_loan_amount"] = np.log1p(loan.clip(lower=0))

        # The extracted date features replace the original date column.
        X = X.drop(columns=[DATE_COLUMN])
        return X


def build_pipeline(X, numeric_imputer):
    """Build the complete feature and preprocessing pipeline."""

    numeric_columns = [
        "age",
        "annual_income",
        "loan_amount",
        "credit_score",
        "application_year",
        "application_month",
        "application_day",
        "income_to_loan_ratio",
        "log_loan_amount",
    ]

    # One-hot encode all remaining text columns except the ordered column.
    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
        and column not in [DATE_COLUMN, ORDINAL_COLUMN]
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", numeric_imputer),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal_encoder",
                OrdinalEncoder(
                    categories=EDUCATION_ORDER,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    # Every imputer and encoder is inside this ColumnTransformer.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
            ("ordered_education", ordinal_pipeline, [ORDINAL_COLUMN]),
        ],
        remainder="drop",
    )

    # Feature engineering, preprocessing, and model are all in one Pipeline.
    return Pipeline(
        steps=[
            ("feature_engineering", LoanFeatureEngineer()),
            ("preprocessing", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {DATA_PATH}\n"
            "Put sample_loan_data.csv beside Task2.py."
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = {
        TARGET_COLUMN,
        DATE_COLUMN,
        ORDINAL_COLUMN,
        "age",
        "annual_income",
        "loan_amount",
        "credit_score",
    }
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing these columns: {sorted(missing)}")

    # Remove rows that do not have a known target value.
    data = data.dropna(subset=[TARGET_COLUMN]).copy()
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    if y.nunique() != 2:
        raise ValueError(f"{TARGET_COLUMN} must contain exactly two classes.")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Compare median imputation against KNN imputation on numeric columns.
    # Each imputer is fitted separately inside each cross-validation fold.
    median_pipeline = build_pipeline(
        X,
        SimpleImputer(strategy="median"),
    )
    median_scores = cross_val_score(
        median_pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )
    print(
        "Median imputer CV accuracy: "
        f"{median_scores.mean():.2f} +/- {median_scores.std():.2f}"
    )

    knn_pipeline = build_pipeline(
        X,
        KNNImputer(n_neighbors=3),
    )
    knn_scores = cross_val_score(
        knn_pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )
    print(
        "KNN imputer CV accuracy:    "
        f"{knn_scores.mean():.2f} +/- {knn_scores.std():.2f}"
    )

    # Fit the final pipeline on all labeled data.
    # Median imputation is used as the final model here.
    final_pipeline = median_pipeline.fit(X, y)

    model_path = Path(__file__).with_name("loan_default_pipeline.joblib")
    joblib.dump(final_pipeline, model_path)
    print(f"Saved fitted pipeline to: {model_path}")

    # Load the saved pipeline and predict from three raw, unprocessed rows.
    loaded_pipeline = joblib.load(model_path)

    new_applicants = pd.DataFrame(
        [
            {
                "age": 30,
                "annual_income": 52000,
                "loan_amount": 12000,
                "credit_score": 700,
                "employment_type": "full_time",
                "loan_purpose": "car",
                "application_date": "2025-01-15",
                "education_level": "bachelor",
            },
            {
                "age": 24,
                "annual_income": 29000,
                "loan_amount": 18000,
                "credit_score": 590,
                "employment_type": "contract",
                "loan_purpose": "medical",
                "application_date": "2025-02-10",
                "education_level": "high_school",
            },
            {
                "age": 45,
                "annual_income": 85000,
                "loan_amount": 20000,
                "credit_score": 750,
                "employment_type": "full_time",
                "loan_purpose": "home",
                "application_date": "2025-03-05",
                "education_level": "master",
            },
        ]
    )

    predictions = loaded_pipeline.predict(new_applicants)
    print("Predictions for 3 new rows (0 = no default, 1 = default):")
    print(predictions)


if __name__ == "__main__":
    main()