from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the same built-in Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Use the same split as Task 1 for a fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train and test a single decision tree
single_tree = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)
single_tree.fit(X_train, y_train)
tree_accuracy = accuracy_score(y_test, single_tree.predict(X_test))

# Train and test a random forest with 200 trees and OOB scoring
forest = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,
    random_state=42
)
forest.fit(X_train, y_train)
forest_accuracy = accuracy_score(y_test, forest.predict(X_test))

print(f"Single tree test accuracy: {tree_accuracy:.2f}")
print(f"Random forest test accuracy: {forest_accuracy:.2f}")
print(f"OOB score: {forest.oob_score_:.2f}")

# Try different numbers of trees
tree_counts = [10, 50, 100, 300]
accuracies = []

for count in tree_counts:
    model = RandomForestClassifier(
        n_estimators=count,
        random_state=42
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    accuracies.append(accuracy)
    print(f"{count} trees: test accuracy = {accuracy:.2f}")

# Plot accuracy for each forest size
plt.figure(figsize=(8, 5))
plt.plot(tree_counts, accuracies, marker="o")
plt.xticks(tree_counts)
plt.xlabel("Number of trees (n_estimators)")
plt.ylabel("Test accuracy")
plt.title("Random Forest Accuracy by Number of Trees")
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig("wine_forest_accuracy.png", dpi=150)
print("\nSaved accuracy plot as wine_forest_accuracy.png")

# Plot feature importances from the 200-tree forest
importances = forest.feature_importances_
sorted_indices = importances.argsort()

plt.figure(figsize=(9, 7))
plt.barh(
    [wine.feature_names[i] for i in sorted_indices],
    importances[sorted_indices]
)
plt.xlabel("Feature importance")
plt.ylabel("Wine feature")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("wine_forest_feature_importances.png", dpi=150)
print("Saved feature-importance plot as wine_forest_feature_importances.png")