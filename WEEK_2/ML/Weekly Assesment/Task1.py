"""Task 1: concepts, train/validation/test splits, overfitting, CV, and leakage.

Before running, set DATA_PATH and TARGET_COLUMN below to match the trainer's CSV.
Run: python task1_concepts_splits_leakage.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ----------- Change these two settings for your trainer's dataset -----------
DATA_PATH = Path("C:/Users/Lenovo/Desktop/Python/WEEK_2/ML/Weekly Assesment/sample_loan_data.csv")
TARGET_COLUMN = "default"  # The binary column to predict
# ---------------------------------------------------------------------------

RANDOM_STATE = 42


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute missing values and one-hot encode text columns inside each model."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numbers", numeric_pipeline, numeric_columns),
            ("categories", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH.resolve()}. Put your CSV there or change DATA_PATH."
        )

    data = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column {TARGET_COLUMN!r} is missing. CSV columns: {list(data.columns)}"
        )

    # Remove rows without a known target; features may still contain missing values.
    data = data.dropna(subset=[TARGET_COLUMN]).copy()
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    if y.nunique() != 2:
        raise ValueError(
            f"This task expects a binary target, but {TARGET_COLUMN!r} has {y.nunique()} classes."
        )
    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after removing the target.")

    # First hold out 20% for final testing. Then split the remaining 80% into
    # 60% train and 20% validation. Stratification keeps class proportions similar.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )

    print("Split sizes (test is held aside and not used below):")
    print(f"  Train:      {len(X_train)} rows ({len(X_train) / len(X):.0%})")
    print(f"  Validation: {len(X_val)} rows ({len(X_val) / len(X):.0%})")
    print(f"  Test:       {len(X_test)} rows ({len(X_test) / len(X):.0%})")

    # 1) Compare tree depth using validation accuracy. The preprocessing is
    # fitted on training data only because it lives inside the Pipeline.
    depths = [1, 3, 5, 10, None]
    train_accuracies = []
    val_accuracies = []
    print("\nDecision tree train and validation accuracy:")
    for depth in depths:
        model = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(X_train)),
                ("tree", DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)),
            ]
        )
        model.fit(X_train, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        print(
            f"  Depth {str(depth):>4}: train {train_accuracy:.2f} | "
            f"validation {val_accuracy:.2f}"
        )

    depth_labels = [str(depth) for depth in depths]
    positions = range(len(depths))
    plt.figure(figsize=(8, 5))
    plt.plot(positions, train_accuracies, marker="o", label="Train accuracy")
    plt.plot(positions, val_accuracies, marker="o", label="Validation accuracy")
    plt.xticks(list(positions), depth_labels)
    plt.xlabel("Decision tree max_depth (None means unlimited)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.title("Decision tree: training vs validation accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = Path("task1_tree_accuracy.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to: {plot_path.resolve()}")
    plt.show()

    # 2) Five-fold CV on the training split only. Each fold gets its own fitted
    # imputer/encoder, avoiding preprocessing leakage between folds.
    logistic_model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(X_train)),
            ("scale", StandardScaler(with_mean=False)),
            ("logistic_regression", LogisticRegression(max_iter=2000)),
        ]
    )
    cv_scores = cross_val_score(
        logistic_model, X_train, y_train, cv=5, scoring="accuracy"
    )
    print(
        f"\nLogistic Regression 5-fold CV accuracy: "
        f"{cv_scores.mean():.2f} +/- {cv_scores.std():.2f} (mean +/- std)"
    )

    # 3) Leakage demonstration using numeric columns. The intentionally wrong
    # version learns the scaler's means/stds from train + validation features.
    # Test remains untouched. Rows with missing numeric values are median-imputed.
    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    print("\nScaling leakage demonstration (numeric columns only):")
    if not numeric_columns:
        print("  Skipped: this dataset has no numeric feature columns to scale.")
    else:
        X_train_num = X_train[numeric_columns]
        X_val_num = X_val[numeric_columns]

        # Deliberately incorrect: fit imputer and scaler using validation features too.
        bad_imputer = SimpleImputer(strategy="median")
        trainval_num = pd.concat([X_train_num, X_val_num], axis=0)
        trainval_imputed = bad_imputer.fit_transform(trainval_num)
        bad_scaler = StandardScaler()
        trainval_scaled = bad_scaler.fit_transform(trainval_imputed)
        bad_train_scaled = trainval_scaled[: len(X_train)]
        bad_val_scaled = trainval_scaled[len(X_train) :]
        bad_model = LogisticRegression(max_iter=2000)
        bad_model.fit(bad_train_scaled, y_train)
        bad_accuracy = accuracy_score(y_val, bad_model.predict(bad_val_scaled))

        # Correct: fit imputer and scaler only on training rows.
        good_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("logistic_regression", LogisticRegression(max_iter=2000)),
            ]
        )
        good_pipeline.fit(X_train_num, y_train)
        good_accuracy = accuracy_score(y_val, good_pipeline.predict(X_val_num))
        print(f"  Wrong (scaled using train + validation): {bad_accuracy:.2f}")
        print(f"  Correct (fit scaler on train only):       {good_accuracy:.2f}")

    print(
        "\nLeakage explanation (4 lines):\n"
        "1. Scaling uses the mean and standard deviation of the data it is fitted on.\n"
        "2. Fitting before splitting lets validation information influence those values.\n"
        "3. This can make validation results look better or otherwise less trustworthy.\n"
        "4. Split first; fit preprocessing on train only, then transform validation/test."
    )

    print(
        "\nProblem types:\n"
        "  Spam detection: supervised (emails have spam/not-spam labels).\n"
        "  Customer segmentation: unsupervised (find groups without supplied labels).\n"
        "  House price prediction: supervised (learn from known sale prices).\n"
        "  Anomaly detection: usually unsupervised (find unusual records without labels)."
    )
    print(
        "\nML workflow in 7 steps:\n"
        "  1. Define the problem, target, and success metric.\n"
        "  2. Collect the data and understand its columns and quality.\n"
        "  3. Clean data and explore patterns, missing values, and class balance.\n"
        "  4. Split into train, validation, and a held-out test set.\n"
        "  5. Build preprocessing and train candidate models on training data.\n"
        "  6. Tune/compare using validation or cross-validation; check for leakage.\n"
        "  7. Evaluate once on test, then document and deploy/monitor the model."
    )


if __name__ == "__main__":
    main()
