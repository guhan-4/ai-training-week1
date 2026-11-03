import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------
# 1. StandardScaler and MinMaxScaler example
# ------------------------------------------------------------

df = pd.DataFrame({
    "age": [25, 30, 40, 35, 50],
    "income": [30000, 45000, 60000, 80000, 52000],
})

# StandardScaler: each feature has mean about 0 and population
# standard deviation about 1 after transformation.
standard_scaler = StandardScaler()
standard_scaled = pd.DataFrame(
    standard_scaler.fit_transform(df[["age", "income"]]),
    columns=["age", "income"],
)

print("StandardScaler results:")
print(standard_scaled)
print("Means:")
print(standard_scaled.mean().round(6))
print("Standard deviations (ddof=0):")
print(standard_scaled.std(ddof=0).round(6))

# MinMaxScaler: each feature is mapped to the range [0, 1].
minmax_scaler = MinMaxScaler()
minmax_scaled = pd.DataFrame(
    minmax_scaler.fit_transform(df[["age", "income"]]),
    columns=["age", "income"],
)

print("\nMinMaxScaler results:")
print(minmax_scaled)
print("Minimum values:")
print(minmax_scaled.min())
print("Maximum values:")
print(minmax_scaled.max())

# ------------------------------------------------------------
# 2. Compare KNN and decision tree on the breast cancer dataset
# ------------------------------------------------------------

data = load_breast_cancer()
X = data.data
y = data.target

# Split before fitting any scaler. Stratification keeps class proportions
# similar in the training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Fit the scaler only on training data, then transform both sets.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN without scaling
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
knn_unscaled_accuracy = accuracy_score(
    y_test,
    knn_unscaled.predict(X_test),
)

# KNN with scaling
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
knn_scaled_accuracy = accuracy_score(
    y_test,
    knn_scaled.predict(X_test_scaled),
)

# Decision tree without scaling
tree_unscaled = DecisionTreeClassifier(random_state=42)
tree_unscaled.fit(X_train, y_train)
tree_unscaled_accuracy = accuracy_score(
    y_test,
    tree_unscaled.predict(X_test),
)

# Decision tree with scaling
tree_scaled = DecisionTreeClassifier(random_state=42)
tree_scaled.fit(X_train_scaled, y_train)
tree_scaled_accuracy = accuracy_score(
    y_test,
    tree_scaled.predict(X_test_scaled),
)

print("\nBreast cancer dataset accuracy:")
print(f"KNN without scaling: {knn_unscaled_accuracy:.2f}")
print(f"KNN with scaling:    {knn_scaled_accuracy:.2f}")
print(f"Decision tree without scaling: {tree_unscaled_accuracy:.2f}")
print(f"Decision tree with scaling:    {tree_scaled_accuracy:.2f}")

if tree_unscaled_accuracy == tree_scaled_accuracy:
    print("Decision tree accuracy is the same in this split.")
else:
    print("Decision tree accuracy differs slightly in this run.")