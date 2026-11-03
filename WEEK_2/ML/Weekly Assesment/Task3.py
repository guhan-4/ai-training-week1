"""Task 3: supervised classifiers, regression, and K-Means segmentation.

Put sample_loan_data.csv beside this script. Install packages with:
    python -m pip install pandas numpy scikit-learn matplotlib xgboost
Run with:
    python Task3.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


DATA_PATH = Path(__file__).with_name("sample_loan_data.csv")
TARGET_COLUMN = "default"
REGRESSION_TARGET = "loan_amount"
RANDOM_STATE = 42


def make_preprocessor(X):
    """Make numeric and categorical preprocessing that is fitted in each pipeline."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def make_classifier_pipeline(X, classifier):
    return Pipeline(
        steps=[
            ("preprocessing", make_preprocessor(X)),
            ("classifier", classifier),
        ]
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {DATA_PATH}\n"
            "Put sample_loan_data.csv in the same folder as Task3.py."
        )

    data = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"CSV must contain target column {TARGET_COLUMN!r}.")
    if REGRESSION_TARGET not in data.columns:
        raise ValueError(f"CSV must contain {REGRESSION_TARGET!r} for regression.")

    data = data.dropna(subset=[TARGET_COLUMN, REGRESSION_TARGET]).copy()
    y = data[TARGET_COLUMN]
    if y.nunique() != 2:
        raise ValueError(f"{TARGET_COLUMN!r} must have exactly two classes.")

    # ------------------------------------------------------------------
    # A) Four supervised classifiers
    # ------------------------------------------------------------------
    # Drop the raw date string to avoid treating every date as its own category.
    X_classification = data.drop(columns=[TARGET_COLUMN, "application_date"], errors="ignore")
    X_train, X_val, y_train, y_val = train_test_split(
        X_classification,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
        ),
    }
    fitted = {}

    print("CLASSIFICATION VALIDATION ACCURACY")
    for name, classifier in models.items():
        pipeline = make_classifier_pipeline(X_train, classifier)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        predictions = pipeline.predict(X_val)
        print(f"{name}: {accuracy_score(y_val, predictions):.3f}")

    forest = fitted["Random Forest"].named_steps["classifier"]
    print(f"Random Forest OOB score: {forest.oob_score_:.3f}")

    # XGBoost early stopping uses the validation set. The preprocessor is first
    # fitted on training rows only, then transforms validation rows for eval_set.
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        early_stopping_rounds=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_pipeline = make_classifier_pipeline(X_train, xgb_model)
    fitted_preprocessor = xgb_pipeline.named_steps["preprocessing"]
    fitted_preprocessor.fit(X_train, y_train)
    X_val_transformed = fitted_preprocessor.transform(X_val)
    xgb_pipeline.fit(
        X_train,
        y_train,
        classifier__eval_set=[(X_val_transformed, y_val)],
        classifier__verbose=False,
    )
    fitted["XGBoost"] = xgb_pipeline
    xgb_predictions = xgb_pipeline.predict(X_val)
    print(f"XGBoost (early stopping): {accuracy_score(y_val, xgb_predictions):.3f}")

    # Logistic probability estimates and predicted labels for five validation rows.
    logistic = fitted["Logistic Regression"]
    five_rows = X_val.iloc[:5]
    five_probabilities = logistic.predict_proba(five_rows)
    five_predictions = logistic.predict(five_rows)
    probability_table = pd.DataFrame(
        five_probabilities,
        columns=[f"P(default={label})" for label in logistic.classes_],
        index=five_rows.index,
    )
    probability_table["predict() class"] = five_predictions
    print("\nLOGISTIC REGRESSION: predict_proba vs predict (5 validation rows)")
    print(probability_table.to_string())
    print("predict_proba gives class probabilities; predict chooses the most likely class.")

    # Show the five largest built-in feature importances for forest and XGBoost.
    print("\nTOP 5 RANDOM FOREST FEATURES")
    forest_names = fitted["Random Forest"].named_steps["preprocessing"].get_feature_names_out()
    forest_importances = fitted["Random Forest"].named_steps["classifier"].feature_importances_
    forest_top = pd.Series(forest_importances, index=forest_names).nlargest(5)
    print(forest_top.to_string())

    print("\nTOP 5 XGBOOST FEATURES")
    xgb_names = fitted["XGBoost"].named_steps["preprocessing"].get_feature_names_out()
    xgb_importances = fitted["XGBoost"].named_steps["classifier"].feature_importances_
    xgb_top = pd.Series(xgb_importances, index=xgb_names).nlargest(5)
    print(xgb_top.to_string())

    # ------------------------------------------------------------------
    # B) Linear regression, Ridge, and Lasso for loan amount
    # ------------------------------------------------------------------
    # Do not use loan_amount itself as an input. Also exclude default because
    # it is a separate prediction target and may not be known at application time.
    X_regression = data.drop(
        columns=[REGRESSION_TARGET, TARGET_COLUMN, "application_date"], errors="ignore"
    )
    y_regression = data[REGRESSION_TARGET]
    regression_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=20000),
    }
    regression_coefficients = {}
    regression_feature_names = None

    print("\nLOAN AMOUNT REGRESSION COEFFICIENTS")
    for name, regressor in regression_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessing", make_preprocessor(X_regression)),
                ("regressor", regressor),
            ]
        )
        pipeline.fit(X_regression, y_regression)
        feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
        coefficients = pipeline.named_steps["regressor"].coef_
        regression_coefficients[name] = coefficients
        regression_feature_names = feature_names

    coefficient_table = pd.DataFrame(
        regression_coefficients,
        index=regression_feature_names,
    )
    print(coefficient_table.round(3).to_string())
    lasso_zero_count = int(np.sum(np.isclose(regression_coefficients["Lasso"], 0.0)))
    print(f"Lasso zero coefficients: {lasso_zero_count} of {len(regression_feature_names)}")

    # ------------------------------------------------------------------
    # C) K-Means customer segments on three numeric columns
    # ------------------------------------------------------------------
    cluster_columns = ["age", "annual_income", "credit_score"]
    missing_cluster_columns = [col for col in cluster_columns if col not in data.columns]
    if missing_cluster_columns:
        raise ValueError(f"CSV needs numeric clustering columns: {missing_cluster_columns}")

    X_cluster = data[cluster_columns]
    cluster_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_cluster_scaled = cluster_preprocessor.fit_transform(X_cluster)

    k_values = range(2, 9)
    inertias = []
    silhouette_scores = []
    print("\nK-MEANS RESULTS")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(X_cluster_scaled)
        score = silhouette_score(X_cluster_scaled, labels)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(score)
        print(f"k={k}: inertia={kmeans.inertia_:.2f}, silhouette={score:.3f}")

    # Save the elbow plot so it can be included in the assessment.
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values), inertias, marker="o")
    plt.xticks(list(k_values))
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.title("K-Means elbow curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = Path(__file__).with_name("task3_kmeans_elbow.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved elbow plot to: {plot_path}")
    plt.show()

    # Describe segments using the k with the best silhouette score.
    best_k = list(k_values)[int(np.argmax(silhouette_scores))]
    best_kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    segment_labels = best_kmeans.fit_predict(X_cluster_scaled)
    segment_summary = data[cluster_columns].copy()
    segment_summary["segment"] = segment_labels
    print(f"\nSEGMENT DESCRIPTIONS (selected k={best_k}, highest silhouette)")
    print("Median profile by segment:")
    print(segment_summary.groupby("segment")[cluster_columns].median().round(1).to_string())
    print("\nNumber of customers in each segment:")
    print(segment_summary["segment"].value_counts().sort_index().to_string())
    print(
        "\nDescribe the groups from their median age, income, and credit score. "
        "For example, a segment with higher income and credit score can be described "
        "as higher-income, stronger-credit customers."
    )


if __name__ == "__main__":
    main()
