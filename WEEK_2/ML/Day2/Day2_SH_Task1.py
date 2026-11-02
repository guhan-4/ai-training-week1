import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# Load the built-in breast cancer dataset
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
feature_names = dataset.feature_names

# Split the data into training, validation, and test sets.
# The test set stays separate until final evaluation.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

# Compare the requested learning rates.
# Smaller learning rates use more possible trees.
settings = [
    {"learning_rate": 0.30, "n_estimators": 300},
    {"learning_rate": 0.10, "n_estimators": 600},
    {"learning_rate": 0.03, "n_estimators": 1500},
]

results = []
models = {}

for setting in settings:
    learning_rate = setting["learning_rate"]

    model = XGBClassifier(
        n_estimators=setting["n_estimators"],
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1
    )

    # The validation set is used for early stopping.
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    # best_iteration is zero-based, so add 1 for the number of trees.
    best_iteration = model.best_iteration + 1

    # Test set is used only for the final accuracy measurement.
    predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)

    results.append({
        "learning_rate": learning_rate,
        "best_iteration": best_iteration,
        "test_accuracy": test_accuracy
    })
    models[learning_rate] = model

# Print the required results table
results_table = pd.DataFrame(results)
print("\nResults:")
print(
    results_table.to_string(
        index=False,
        formatters={"test_accuracy": "{:.3f}".format}
    )
)

# Pick the model with the highest test accuracy for feature-importance display.
# The test scores are for comparison; do not tune further based on them.
best_row = max(results, key=lambda row: row["test_accuracy"])
best_model = models[best_row["learning_rate"]]

print(
    f"\nFeature importances for learning_rate="
    f"{best_row['learning_rate']}:"
)

# Print the 10 most important features
importances = best_model.feature_importances_
top_indices = importances.argsort()[::-1][:10]

for rank, index in enumerate(top_indices, start=1):
    print(f"{rank}. {feature_names[index]}: {importances[index]:.4f}")

# Plot the 10 most important features
plt.figure(figsize=(9, 6))
plt.barh(
    [feature_names[i] for i in top_indices][::-1],
    [importances[i] for i in top_indices][::-1]
)
plt.xlabel("Importance")
plt.title("Top 10 XGBoost Feature Importances")
plt.tight_layout()
plt.savefig("breast_cancer_xgboost_importance.png", dpi=150)
print("\nSaved plot as breast_cancer_xgboost_importance.png")