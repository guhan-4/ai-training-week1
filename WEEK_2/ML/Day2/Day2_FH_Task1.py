from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the built-in wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Keep some data aside for a final test.
# Stratify keeps the wine classes represented in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 1. Compare Gini and entropy using the same train/test split
print("Test accuracy by criterion:")
for criterion in ["gini", "entropy"]:
    model = DecisionTreeClassifier(
        criterion=criterion,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{criterion}: {accuracy:.3f}")

# 2. Find the best max_depth and min_samples_leaf combination.
# GridSearchCV chooses settings using cross-validation on the training data.
parameter_grid = {
    "max_depth": [2, 3, 4, 5, 6, None],
    "min_samples_leaf": [1, 2, 3, 4, 5],
}

search = GridSearchCV(
    estimator=DecisionTreeClassifier(criterion="gini", random_state=42),
    param_grid=parameter_grid,
    cv=5,
    scoring="accuracy"
)
search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest settings (selected using training data):")
print("max_depth:", search.best_params_["max_depth"])
print("min_samples_leaf:", search.best_params_["min_samples_leaf"])
print(f"Cross-validation accuracy: {search.best_score_:.3f}")

# Evaluate the selected model once on the held-out test data
test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
print(f"Best model test accuracy: {test_accuracy:.3f}")

# 3. Plot the first 3 levels of the tuned tree
plt.figure(figsize=(20, 10))
plot_tree(
    best_model,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)
plt.title("Wine Decision Tree (shown to depth 3)")
plt.tight_layout()
plt.savefig("wine_decision_tree.png", dpi=150)
print("\nSaved tree plot as wine_decision_tree.png")

# Explain the first two splits using the actual tree rules
tree = best_model.tree_

def describe_split(node_id, description):
    feature_id = tree.feature[node_id]
    threshold = tree.threshold[node_id]
    feature_name = wine.feature_names[feature_id]

    print(
        f"{description}: The tree checks whether "
        f"{feature_name} is <= {threshold:.3f}."
    )

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    # A child is a leaf if it has no further split.
    if tree.feature[left_child] != -2:
        describe_split(left_child, "Next split on the left branch")
    if tree.feature[right_child] != -2:
        describe_split(right_child, "Next split on the right branch")

print("\nFirst split and the next split on each branch:")
describe_split(0, "First split")

# 4. Print the five most important features from the tuned tree
print("\nTop 5 feature importances:")
importances = best_model.feature_importances_
top_five = importances.argsort()[::-1][:5]

for rank, feature_id in enumerate(top_five, start=1):
    print(
        f"{rank}. {wine.feature_names[feature_id]}: "
        f"{importances[feature_id]:.3f}"
    )