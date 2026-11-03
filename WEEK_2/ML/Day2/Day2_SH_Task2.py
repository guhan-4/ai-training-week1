import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Part 1: Make 300 points with 4 natural groups
# ------------------------------------------------------------

X, _ = make_blobs(
    n_samples=300,
    centers=4,
    random_state=7
)

k_values = range(2, 9)
inertias = []
silhouette_scores = []

# Try K-Means with k = 2, 3, ..., 8
for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=7,
        n_init=10
    )
    labels = model.fit_predict(X)

    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

# Pick the k with the highest silhouette score
best_index = silhouette_scores.index(max(silhouette_scores))
best_k = list(k_values)[best_index]
best_score = silhouette_scores[best_index]

print("Results for generated points:")
for k, inertia, score in zip(k_values, inertias, silhouette_scores):
    print(f"k={k}: inertia={inertia:.2f}, silhouette={score:.3f}")

print(f"\nBest k by silhouette: {best_k} (score {best_score:.3f})")

# Plot the elbow curve (inertia)
plt.figure(figsize=(7, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xticks(list(k_values))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("kmeans_elbow.png", dpi=150)

# Fit final model using the silhouette-selected k
final_model = KMeans(
    n_clusters=best_k,
    random_state=7,
    n_init=10
)
final_labels = final_model.fit_predict(X)

# Plot clusters and centroids
plt.figure(figsize=(7, 5))
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=final_labels,
    cmap="viridis",
    s=35,
    label="Data points"
)
plt.scatter(
    final_model.cluster_centers_[:, 0],
    final_model.cluster_centers_[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroids"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"K-Means Clusters (k={best_k})")
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_final_clusters.png", dpi=150)


# ------------------------------------------------------------
# Part 2: Mall Customers style income/spending segmentation
# This creates example data; it is not real customer data.
# ------------------------------------------------------------

# Five example customer types:
# lower income/lower spending, lower/higher, middle/middle,
# higher/lower, and higher/higher.
customer_centers = [
    [20, 20],
    [20, 80],
    [50, 50],
    [80, 20],
    [80, 80],
]

customer_data, _ = make_blobs(
    n_samples=300,
    centers=customer_centers,
    cluster_std=8,
    random_state=7
)

# Keep the example values within plausible display ranges:
# income is in thousands of dollars; spending score is 1–100.
customer_data[:, 0] = customer_data[:, 0].clip(10, 140)
customer_data[:, 1] = customer_data[:, 1].clip(1, 100)

customers = pd.DataFrame(
    customer_data,
    columns=["Annual Income (k$)", "Spending Score"]
)

# Scale both columns before clustering so they contribute comparably.
scaler = StandardScaler()
customer_scaled = scaler.fit_transform(customers)

customer_kmeans = KMeans(
    n_clusters=5,
    random_state=7,
    n_init=10
)
customers["Cluster"] = customer_kmeans.fit_predict(customer_scaled)

# Convert cluster centers back to the original income and score units.
centers_original = scaler.inverse_transform(customer_kmeans.cluster_centers_)
centers = pd.DataFrame(
    centers_original,
    columns=["Annual Income (k$)", "Spending Score"]
)

print("\nMall Customers style segments (synthetic example):")

income_middle = customers["Annual Income (k$)"].median()
spending_middle = customers["Spending Score"].median()

for cluster_id, center in centers.iterrows():
    income = center["Annual Income (k$)"]
    spending = center["Spending Score"]

    income_label = "higher-income" if income >= income_middle else "lower-income"
    spending_label = "higher-spending" if spending >= spending_middle else "lower-spending"

    print(
        f"Cluster {cluster_id}: average income ${income:.1f}k, "
        f"average spending score {spending:.1f} — "
        f"{income_label} customers with {spending_label}."
    )

# Plot the customer segments and their centers.
plt.figure(figsize=(8, 6))
plt.scatter(
    customers["Annual Income (k$)"],
    customers["Spending Score"],
    c=customers["Cluster"],
    cmap="tab10",
    s=35
)
plt.scatter(
    centers["Annual Income (k$)"],
    centers["Spending Score"],
    c="black",
    marker="X",
    s=220,
    label="Centroids"
)
plt.xlabel("Annual Income (thousands of dollars)")
plt.ylabel("Spending Score")
plt.title("Mall Customers Style Segments (Synthetic Data)")
plt.legend()
plt.tight_layout()
plt.savefig("mall_customer_segments.png", dpi=150)

print("\nSaved plots:")
print("- kmeans_elbow.png")
print("- kmeans_final_clusters.png")
print("- mall_customer_segments.png")

# Uncomment to show the plots in windows when running locally:
# plt.show()