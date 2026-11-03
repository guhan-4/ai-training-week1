"""Task 4: evaluation, threshold selection, hyperparameter tuning, and final test.

Put sample_loan_data.csv beside this script. Install packages with:
    python -m pip install pandas numpy scikit-learn matplotlib xgboost
Run with:
    python Task4.py

The test split is held aside until the final evaluation section.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


DATA_PATH = Path(__file__).with_name("sample_loan_data.csv")
TARGET_COLUMN = "default"
REGRESSION_TARGET = "loan_amount"
RANDOM_STATE = 42


def make_preprocessor(X):
    """Build numeric and categorical preprocessing for use inside a Pipeline."""
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


def make_pipeline(X, estimator, final_step_name):
    return Pipeline(
        steps=[
            ("preprocessing", make_preprocessor(X)),
            (final_step_name, estimator),
        ]
    )


def positive_class_probabilities(model, X):
    """Return P(class=1), regardless of the order in model.classes_."""
    positive_index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, positive_index]


def print_classification_metrics(name, y_true, predictions, probabilities):
    print(f"\n{name}")
    print(f"  Accuracy:  {accuracy_score(y_true, predictions):.3f}")
    print(f"  Precision: {precision_score(y_true, predictions, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_true, predictions, zero_division=0):.3f}")
    print(f"  F1:        {f1_score(y_true, predictions, zero_division=0):.3f}")
    print("  Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))
    print(f"  ROC-AUC:   {roc_auc_score(y_true, probabilities):.3f}")


def regression_metrics(y_true, predictions):
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": rmse,
        "R-squared": r2_score(y_true, predictions),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {DATA_PATH}\n"
            "Put sample_loan_data.csv in the same folder as Task4.py."
        )

    data = pd.read_csv(DATA_PATH)
    required = {TARGET_COLUMN, REGRESSION_TARGET}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    data = data.dropna(subset=[TARGET_COLUMN, REGRESSION_TARGET]).copy()
    y = data[TARGET_COLUMN]
    if set(pd.unique(y)) != {0, 1}:
        raise ValueError("This script expects the default target encoded as 0 and 1.")

    # Make a final 20% test holdout. The remaining 80% is split 75/25 into
    # train and validation, giving an overall 60/20/20 stratified split.
    X_all = data.drop(columns=[TARGET_COLUMN, REGRESSION_TARGET, "application_date"], errors="ignore")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )
    print(f"Rows: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")
    print("The test split is reserved for the final evaluation below.")

    # All search preprocessing lives inside each estimator Pipeline, so it is
    # refitted on the training fold during cross-validation.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1) Grid search for Random Forest using exactly three parameters.
    rf_pipeline = make_pipeline(
        X_train,
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "classifier",
    )
    rf_grid = {
        "classifier__n_estimators": [100, 250],
        "classifier__max_depth": [None, 4, 8],
        "classifier__min_samples_split": [2, 5],
    }
    rf_search = GridSearchCV(
        rf_pipeline,
        param_grid=rf_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    rf_search.fit(X_train, y_train)
    print("\nRANDOM FOREST GRID SEARCH")
    print(f"Best parameters: {rf_search.best_params_}")
    print(f"Best mean CV F1: {rf_search.best_score_:.3f}")

    # 2) Randomized search for XGBoost with 15 sampled parameter combinations.
    xgb_pipeline = make_pipeline(
        X_train,
        XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "classifier",
    )
    xgb_distributions = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [2, 3, 4, 5],
        "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.9, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=xgb_distributions,
        n_iter=15,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    xgb_search.fit(X_train, y_train)
    print("\nXGBOOST RANDOMIZED SEARCH")
    print(f"Best parameters: {xgb_search.best_params_}")
    print(f"Best mean CV F1: {xgb_search.best_score_:.3f}")

    # Fit the other classifier pipelines on train only.
    logistic = make_pipeline(
        X_train, LogisticRegression(max_iter=2000), "classifier"
    ).fit(X_train, y_train)
    decision_tree = make_pipeline(
        X_train,
        DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "classifier",
    ).fit(X_train, y_train)

    # Use the tuned estimators for RF and XGBoost.
    classifiers = {
        "Logistic Regression": logistic,
        "Decision Tree": decision_tree,
        "Tuned Random Forest": rf_search.best_estimator_,
        "Tuned XGBoost": xgb_search.best_estimator_,
    }

    # 3) Validation metrics for every classifier. Validation guides model
    # selection and threshold choice; it is not the final test set.
    val_probabilities = {}
    val_aucs = {}
    val_f1s = {}
    print("\nVALIDATION METRICS FOR ALL CLASSIFIERS")
    for name, model in classifiers.items():
        probabilities = positive_class_probabilities(model, X_val)
        predictions = (probabilities >= 0.5).astype(int)
        print_classification_metrics(name, y_val, predictions, probabilities)
        val_probabilities[name] = probabilities
        val_aucs[name] = roc_auc_score(y_val, probabilities)
        val_f1s[name] = f1_score(y_val, predictions, zero_division=0)

    print(
        "\nWhy accuracy can mislead: with imbalanced classes, a model can predict "
        "only the majority class and still have high accuracy while missing most "
        "defaults. Precision, recall, F1, and the confusion matrix reveal that behavior."
    )

    # 4) Plot ROC curves for three selected models on the same validation rows.
    roc_models = ["Logistic Regression", "Tuned Random Forest", "Tuned XGBoost"]
    plt.figure(figsize=(8, 6))
    for name in roc_models:
        fpr, tpr, _ = roc_curve(y_val, val_probabilities[name], pos_label=1)
        auc = val_aucs[name]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (recall)")
    plt.title("Validation ROC curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = Path(__file__).with_name("task4_roc_curves.png")
    plt.savefig(roc_path, dpi=150)
    print(f"\nSaved ROC plot to: {roc_path}")
    plt.show()

    # Pick the classifier with the best validation ROC-AUC; use F1 as tie-breaker.
    best_name = max(classifiers, key=lambda name: (val_aucs[name], val_f1s[name]))
    best_model = classifiers[best_name]
    print(
        f"\nBest model selected on validation: {best_name} | "
        f"ROC-AUC {val_aucs[best_name]:.3f} | F1 {val_f1s[best_name]:.3f}"
    )

    # 5) Find the threshold with the highest precision among thresholds whose
    # validation recall is at least 75%. Threshold 0 predicts all rows positive.
    best_probabilities = val_probabilities[best_name]
    candidate_thresholds = np.unique(np.append(best_probabilities, 0.0))
    acceptable = []
    for threshold in candidate_thresholds:
        candidate_predictions = (best_probabilities >= threshold).astype(int)
        candidate_recall = recall_score(y_val, candidate_predictions, zero_division=0)
        if candidate_recall >= 0.75:
            candidate_precision = precision_score(
                y_val, candidate_predictions, zero_division=0
            )
            acceptable.append((candidate_precision, threshold, candidate_recall))

    # At threshold 0 recall is 1 when both classes are present, so a choice exists.
    chosen_precision, chosen_threshold, chosen_recall = max(
        acceptable, key=lambda row: (row[0], row[1])
    )
    print("\nRECALL-FOCUSED DECISION THRESHOLD (validation set)")
    print(f"Threshold: {chosen_threshold:.4f}")
    print(f"Recall:    {chosen_recall:.3f} (requirement: at least 0.750)")
    print(f"Precision: {chosen_precision:.3f}")

    # 6) Regression: compare Linear, Ridge, and Lasso with a mean baseline.
    # Use validation MAE to choose one regression model. Test stays untouched.
    X_regression = data.drop(
        columns=[REGRESSION_TARGET, TARGET_COLUMN, "application_date"], errors="ignore"
    )
    y_regression = data[REGRESSION_TARGET]
    X_reg_trainval, X_reg_test, y_reg_trainval, y_reg_test = train_test_split(
        X_regression,
        y_regression,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
    X_reg_train, X_reg_val, y_reg_train, y_reg_val = train_test_split(
        X_reg_trainval,
        y_reg_trainval,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )
    regression_estimators = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=20000),
    }
    regression_pipelines = {}
    val_mae = {}
    for name, estimator in regression_estimators.items():
        model = make_pipeline(X_reg_train, estimator, "regressor")
        model.fit(X_reg_train, y_reg_train)
        regression_pipelines[name] = model
        val_mae[name] = mean_absolute_error(y_reg_val, model.predict(X_reg_val))

    selected_regression_name = min(val_mae, key=val_mae.get)
    print("\nREGRESSION MODEL SELECTION (validation MAE)")
    for name, mae in val_mae.items():
        print(f"{name}: validation MAE={mae:.2f}")
    print(f"Selected regression model: {selected_regression_name}")

    # 7) Final test evaluation: evaluate the selected classifier once at the
    # chosen threshold, and the selected regressor against a train-mean baseline.
    print("\nFINAL TEST EVALUATION (held-out test split)")
    test_probabilities = positive_class_probabilities(best_model, X_test)
    test_predictions = (test_probabilities >= chosen_threshold).astype(int)
    print_classification_metrics(
        f"{best_name} at threshold {chosen_threshold:.4f}",
        y_test,
        test_predictions,
        test_probabilities,
    )

    selected_regression = regression_pipelines[selected_regression_name]
    regression_test_predictions = selected_regression.predict(X_reg_test)
    baseline_test_predictions = np.full(
        shape=len(y_reg_test), fill_value=y_reg_train.mean(), dtype=float
    )
    model_scores = regression_metrics(y_reg_test, regression_test_predictions)
    baseline_scores = regression_metrics(y_reg_test, baseline_test_predictions)
    print(f"\nRegression test results: {selected_regression_name}")
    print(
        f"  Model   MAE={model_scores['MAE']:.2f}, "
        f"RMSE={model_scores['RMSE']:.2f}, R-squared={model_scores['R-squared']:.3f}"
    )
    print(
        f"  Baseline MAE={baseline_scores['MAE']:.2f}, "
        f"RMSE={baseline_scores['RMSE']:.2f}, R-squared={baseline_scores['R-squared']:.3f}"
    )
    print("Baseline predicts the mean loan amount from the training rows for every test row.")


if __name__ == "__main__":
    main()
