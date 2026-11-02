# churn_pipeline.py
# Run:
#   pip install pandas numpy matplotlib seaborn scikit-learn joblib
#   python churn_pipeline.py
# Or use your own CSV:
#   python churn_pipeline.py --data path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv

import argparse
import json
import urllib.request
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Churn"


def load_data(path: Path) -> pd.DataFrame:
    """Load a local CSV, or download the IBM sample if it is missing."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path)


def audit_training_data(df: pd.DataFrame) -> None:
    """Check shape, types, missing values, duplicates, and class balance."""
    print("\n=== Training data checks ===")
    print("Shape:", df.shape)
    print("\nColumn types:\n", df.dtypes.to_string())

    missing = df.isna().sum()
    print("\nMissing values:\n", missing[missing > 0].to_string() or "None")

    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nTarget counts:\n", df[TARGET].value_counts().to_string())
    print(
        "\nTarget proportions:\n",
        df[TARGET].value_counts(normalize=True).to_string(),
    )


def make_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """Save five charts relating churn to key features, using training rows only."""
    sns.set_theme(style="whitegrid")

    # Three categorical feature charts
    for feature in ["Contract", "InternetService", "PaymentMethod"]:
        churn_rate = (
            df.groupby(feature, dropna=False)[TARGET]
            .mean()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(
            x=churn_rate.index.astype(str),
            y=churn_rate.values,
            color="#3977a8",
            ax=ax,
        )
        ax.set_title(f"Churn rate by {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Churn rate")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output_dir / f"eda_churn_by_{feature.lower()}.png", dpi=160)
        plt.close(fig)

    # Churn by tenure band
    tenure_data = df[["tenure", TARGET]].copy()
    tenure_data["tenure_band"] = pd.cut(
        tenure_data["tenure"],
        bins=[0, 12, 24, 36, 48, 60, 72],
        labels=["1–12", "13–24", "25–36", "37–48", "49–60", "61–72"],
        include_lowest=True,
    )
    tenure_rate = tenure_data.groupby("tenure_band", observed=False)[TARGET].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=tenure_rate.index.astype(str), y=tenure_rate.values, color="#d17a35", ax=ax)
    ax.set_title("Churn rate by tenure band")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Churn rate")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_churn_by_tenure.png", dpi=160)
    plt.close(fig)

    # Churn by monthly charge quantile
    charge_data = df[["MonthlyCharges", TARGET]].copy()
    charge_data["charge_band"] = pd.qcut(
        charge_data["MonthlyCharges"], q=6, duplicates="drop"
    )
    charge_rate = charge_data.groupby("charge_band", observed=False)[TARGET].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        x=charge_rate.index.astype(str),
        y=charge_rate.values,
        color="#d17a35",
        ax=ax,
    )
    ax.set_title("Churn rate by monthly charge range")
    ax.set_xlabel("Monthly charges")
    ax.set_ylabel("Churn rate")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_churn_by_monthlycharges.png", dpi=160)
    plt.close(fig)


def build_model(X: pd.DataFrame) -> Pipeline:
    """Build preprocessing and classifier as one leakage-safe pipeline."""
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_data(args.data)
    if TARGET not in raw.columns:
        raise ValueError(f"Expected a '{TARGET}' column in the dataset.")

    print("Loaded dataset shape:", raw.shape)

    # Convert blank TotalCharges values to missing numeric values.
    if "TotalCharges" in raw.columns:
        raw["TotalCharges"] = pd.to_numeric(
            raw["TotalCharges"].replace(r"^\s*$", np.nan, regex=True),
            errors="coerce",
        )

    # Convert the target to 0/1.
    target_values = raw[TARGET]
    if set(target_values.dropna().unique()).issubset({"Yes", "No"}):
        y = target_values.map({"Yes": 1, "No": 0})
    elif set(target_values.dropna().unique()).issubset({0, 1}):
        y = target_values.astype(float)
    else:
        raise ValueError("Churn must contain only 'Yes'/'No' or 0/1 values.")

    X = raw.drop(columns=[TARGET, "customerID"], errors="ignore")

    # Split immediately. Do not use the test set for audit, EDA, or model selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y.astype(int),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    training_data = X_train.copy()
    training_data[TARGET] = y_train
    audit_training_data(training_data)
    make_eda(training_data, args.output_dir)

    # Use a validation split from training data to select the F1 threshold.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    selection_model = build_model(X_fit)
    selection_model.fit(X_fit, y_fit)
    validation_probabilities = selection_model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.01)
    validation_f1_scores = [
        f1_score(y_val, validation_probabilities >= threshold)
        for threshold in thresholds
    ]
    best_threshold = float(thresholds[int(np.argmax(validation_f1_scores))])

    print(f"\nSelected threshold from training validation data: {best_threshold:.2f}")
    print(f"Validation F1: {max(validation_f1_scores):.4f}")

    # Refit on all training rows. Evaluate the held-out test set only at this point.
    final_model = build_model(X_train)
    final_model.fit(X_train, y_train)

    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= best_threshold).astype(int)

    test_f1 = f1_score(y_test, test_predictions)
    test_roc_auc = roc_auc_score(y_test, test_probabilities)

    print("\n=== Final test results ===")
    print(f"F1:      {test_f1:.4f}")
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, test_predictions))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["No churn", "Churn"],
        )
    )

    # Save fitted model, threshold, predictions, and run summary.
    joblib.dump(
        {
            "model": final_model,
            "threshold": best_threshold,
            "feature_columns": X.columns.tolist(),
        },
        args.output_dir / "churn_model.joblib",
    )

    pd.DataFrame(
        {
            "actual_churn": y_test.to_numpy(),
            "predicted_churn": test_predictions,
            "churn_probability": test_probabilities,
        },
        index=y_test.index,
    ).to_csv(args.output_dir / "test_predictions.csv", index_label="row_index")

    summary = {
        "dataset": str(args.data),
        "rows": len(raw),
        "test_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "optimization_metric": "F1",
        "selected_threshold": best_threshold,
        "test_f1": float(test_f1),
        "test_roc_auc": float(test_roc_auc),
        "assumptions": [
            "Telco Customer Churn data with Yes/No or 1/0 target.",
            "customerID is an identifier and is excluded from predictors.",
            "Blank TotalCharges values are missing and are median-imputed.",
            "Rows are independent and a stratified random split is appropriate.",
            "F1 is optimized because the classes are imbalanced and both missed churners and false alerts matter.",
        ],
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nArtifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()