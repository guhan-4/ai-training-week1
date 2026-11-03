# Loan Default ML Assessment

This project demonstrates a four-task machine learning workflow for predicting loan default, estimating loan amounts, and exploring customer segments.

## Dataset

`sample_loan_data.csv` is a small **synthetic practice dataset** created for this code. It is not real applicant data. If the trainer supplied a dataset, use that dataset for the assessment results and update each script's data path and column names as needed.

The binary target is `default`:

- `0` = no default
- `1` = default

The sample CSV also has `loan_amount` as the regression target, `application_date` for date features, and `education_level` as an ordered category.

## Setup

Install the packages used by the scripts:

```powershell
python -m pip install pandas numpy scikit-learn matplotlib joblib xgboost
```

Keep the scripts and `sample_loan_data.csv` together in the assessment folder. Task 2, Task 3, and Task 4 load the CSV from the script's folder. Check the `DATA_PATH` setting in Task 1 before running it; it may need to be changed to the CSV's location.

## Run the tasks

From PowerShell, change to the assessment folder and run:

```powershell
python .\Task1.py
python .\Task2.py
python .\Task3.py
python .\Task4.py
```

If your files use different names, replace the script names in these commands.

## Task summary and decisions

### Task 1: Concepts, splits, and leakage

- Creates stratified train, validation, and test splits using random state 42.
- Compares training and validation accuracy for decision trees with several maximum depths.
- Runs 5-fold cross-validation for logistic regression.
- Demonstrates why preprocessing must be fitted on training data only.
- Includes the supervised/unsupervised classifications and a seven-step ML workflow.

### Task 2: Preprocessing, features, and pipeline

- Imputes numeric values with the median and categorical values with the most frequent value; compares numeric KNN imputation.
- One-hot encodes unordered categories and ordinal encodes `education_level`.
- Creates date components, an income-to-loan ratio, and a log loan amount feature.
- Keeps feature engineering and preprocessing inside the model pipeline during cross-validation.
- Saves `loan_default_pipeline.joblib`, reloads it, and predicts for three raw applicant rows.

### Task 3: Supervised and unsupervised models

- Compares Logistic Regression, Decision Tree, Random Forest, and XGBoost classifiers.
- Reports Random Forest out-of-bag score, probabilities, and top feature importances.
- Compares Linear Regression, Ridge, and Lasso coefficients for loan amount.
- Runs K-Means for k values 2 through 8, reports silhouette scores, and saves an elbow plot.

### Task 4: Evaluation and tuning

- Reports accuracy, precision, recall, F1, and confusion matrices for classifiers.
- Tunes Random Forest with GridSearchCV and XGBoost with RandomizedSearchCV.
- Selects a classification threshold using validation recall, targeting at least 75%.
- Compares a regression model with a training-mean baseline and reserves the test data for final evaluation.

## Generated outputs

Depending on which scripts are run, the folder may contain:

- `task1_tree_accuracy.png`
- `loan_default_pipeline.joblib`
- `task3_kmeans_elbow.png`
- `task4_roc_curves.png`

The sample dataset is deliberately small, so scores can vary substantially with the split and should not be treated as reliable estimates of real-world performance.