import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

# Create the DataFrame with missing values
df = pd.DataFrame({
    "age": [25, None, 40, 35, None],
    "income": [30000, 45000, None, 80000, 52000],
    "city": ["Chennai", "Mumbai", None, "Chennai", "Delhi"],
    "size": ["S", "L", "M", "S", "L"],
})

# Add a sample target column because the supplied data has no target.
# Replace these example labels with your actual target values.
df["target"] = ["No", "Yes", "Yes", "No", "Yes"]

# Impute age with the median
age_imputer = SimpleImputer(strategy="median")
df[["age"]] = age_imputer.fit_transform(df[["age"]])

# Impute city with the most frequent value
city_imputer = SimpleImputer(strategy="most_frequent")
df[["city"]] = city_imputer.fit_transform(df[["city"]])

# Compare numeric imputation methods on age and income.
# KNNImputer uses the other numeric column to estimate missing values.
numeric_cols = ["age", "income"]

median_imputer = SimpleImputer(strategy="median")
numeric_median = pd.DataFrame(
    median_imputer.fit_transform(df[numeric_cols]),
    columns=numeric_cols,
    index=df.index,
)

knn_imputer = KNNImputer(n_neighbors=2)
numeric_knn = pd.DataFrame(
    knn_imputer.fit_transform(df[numeric_cols]),
    columns=numeric_cols,
    index=df.index,
)

print("Numeric columns with median imputation:")
print(numeric_median)
print("\nNumeric columns with KNN imputation:")
print(numeric_knn)

# Use median-imputed numeric columns in the final DataFrame
df[numeric_cols] = numeric_median

# One-hot encode city
city_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
city_encoded = pd.DataFrame(
    city_encoder.fit_transform(df[["city"]]),
    columns=city_encoder.get_feature_names_out(["city"]),
    index=df.index,
)

# Ordinal-encode size in the specified order: S < M < L
size_encoder = OrdinalEncoder(
    categories=[["S", "M", "L"]],
    handle_unknown="error",
)
df[["size"]] = size_encoder.fit_transform(df[["size"]])

# Label-encode the target (labels only, not input features)
target_encoder = LabelEncoder()
df["target"] = target_encoder.fit_transform(df["target"])

# Assemble and print the final encoded DataFrame
final_df = pd.concat(
    [df.drop(columns=["city"]), city_encoded],
    axis=1,
)

print("\nFinal encoded DataFrame:")
print(final_df)