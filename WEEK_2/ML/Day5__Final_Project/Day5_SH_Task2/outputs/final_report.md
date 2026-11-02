# Customer churn prediction — final report

## Problem
Identify customers at risk of cancelling so the retention team can prioritize outreach. The operational objective is to catch at least 75% of churners, while using the highest threshold that meets that recall target on a training-only validation set.

## Data
IBM Telco Customer Churn sample: 7,043 customer records and 20 input columns. The `Churn` label is the target. `customerID` is excluded as an identifier; blank `TotalCharges` values are treated as missing and imputed within the model pipeline. Rows are assumed independent. A stratified 80/20 train/test split was made before model selection; the test set was evaluated only once at the end.

## Method
Compared Logistic Regression, Random Forest, and XGBoost. Used five-fold stratified CV on training data, selected the top two by CV ROC-AUC, and tuned them with `RandomizedSearchCV(scoring='roc_auc')`. Selected the final candidate using an internal training validation split. XGBoost used `scale_pos_weight` to account for class imbalance. Preprocessing (median/mode imputation, scaling and one-hot encoding) is fitted inside each pipeline fold. The final classifier was refit on all 80% training rows.

## Results
Chosen model: **Random Forest**. Validation-selected threshold: **0.5267**. On the untouched test set: ROC-AUC **0.8428**, F1 **0.6341**, recall **0.7647**. Confusion matrix (actual rows No/Yes; predicted columns No/Yes): `[[793, 242], [88, 286]]`.

## Limitations
This is a single public benchmark snapshot; results may not generalize to another provider, period, or customer population. A random split does not test future-period drift. The recall target is a proxy for business value: this model does not include outreach cost, offer cost, customer lifetime value, or intervention success. Feature importance and model scores are predictive associations, not causal explanations.

## Recommendations
Pilot the scored list with a capped retention campaign, monitor contact capacity and offer acceptance, and compare incremental retention against a randomized control group. Review precision and campaign cost alongside recall. Revalidate on a recent time-based holdout before production use, monitor drift, and retrain when performance changes. Do not treat a risk score as a reason to deny service or price unfairly.
