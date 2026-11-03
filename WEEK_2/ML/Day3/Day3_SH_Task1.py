import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproducible sample data
rng = np.random.default_rng(42)
n_customers = 200

signup_start = pd.Timestamp("2021-01-01")
signup_end = pd.Timestamp("2024-12-31")
signup_days = (signup_end - signup_start).days

signup_date = signup_start + pd.to_timedelta(
    rng.integers(0, signup_days + 1, size=n_customers),
    unit="D",
)

# Ensure last_purchase_date is on or after signup_date.
days_as_customer = (pd.Timestamp("2025-01-01") - signup_date).days
days_since_purchase = np.array([
    rng.integers(0, days + 1) for days in days_as_customer
])
last_purchase_date = pd.Timestamp("2025-01-01") - pd.to_timedelta(
    days_since_purchase, unit="D"
)

# Right-skewed spending values and positive visit counts
total_spend = rng.lognormal(mean=6.5, sigma=0.8, size=n_customers).round(2)
visits = rng.integers(1, 31, size=n_customers)

df = pd.DataFrame({
    "signup_date": signup_date,
    "last_purchase_date": last_purchase_date,
    "total_spend": total_spend,
    "visits": visits,
})

# Create a sample churn target.
# Recent purchasing and higher spend/visit lower the simulated churn probability.
df["days_since_last_purchase"] = (
    pd.Timestamp("2025-01-01") - df["last_purchase_date"]
).dt.days

churn_probability = (
    0.15
    + 0.55 * (df["days_since_last_purchase"] / 365).clip(0, 1)
    - 0.15 * (df["total_spend"] / df["visits"] > 100)
).clip(0.05, 0.85)

df["churned"] = rng.binomial(1, churn_probability)

# Feature engineering
df["signup_year"] = df["signup_date"].dt.year
df["signup_month"] = df["signup_date"].dt.month
df["signup_weekday"] = df["signup_date"].dt.dayofweek  # Monday=0, Sunday=6
df["spend_per_visit"] = df["total_spend"] / df["visits"]
df["log_total_spend"] = np.log1p(df["total_spend"])

# Plot total_spend distributions before and after log1p
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["total_spend"], bins=25, edgecolor="black")
axes[0].set_title("Total spend before log1p")
axes[0].set_xlabel("Total spend")
axes[0].set_ylabel("Customer count")

axes[1].hist(df["log_total_spend"], bins=25, edgecolor="black")
axes[1].set_title("Total spend after log1p")
axes[1].set_xlabel("log1p(total_spend)")
axes[1].set_ylabel("Customer count")

plt.tight_layout()
plt.show()

# Correlations of engineered features with churned.
# The target is binary (0/1), so Pearson correlation is equivalent to
# the point-biserial correlation for each numeric feature.
new_features = [
    "signup_year",
    "signup_month",
    "signup_weekday",
    "days_since_last_purchase",
    "spend_per_visit",
    "log_total_spend",
]

correlations = (
    df[new_features + ["churned"]]
    .corr(numeric_only=True)["churned"]
    .drop("churned")
    .sort_values(key=lambda values: values.abs(), ascending=False)
)

print("Correlation of each new feature with churned:")
print(correlations.round(3))

print("\nSample of the final dataset:")
print(df.head())