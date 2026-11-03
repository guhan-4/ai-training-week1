import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load a classification dataset
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target

# Add a categorical column for the ColumnTransformer example.
# In a real project, use an actual categorical feature from your data.
X["region"] = ["North", "South", "West", "East"] * (len(X) // 4) + [
    ["North", "South", "West", "East"][i % 4]
    for i in range(len(X) % 4)
]

numeric_columns = X.select_dtypes(include="number").columns.tolist()
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

# Numeric features: impute missing values, then scale
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical features: impute missing values, then one-hot encode
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

# Preprocessing and classifier are in one pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000)),
])

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print(f"Pipeline CV accuracy: {scores.mean():.2f} +/- {scores.std():.2f}")

# No leakage: cross_val_score clones and fits the entire pipeline separately
# in each fold. Imputers, scaler, and encoder learn their values from that
# fold's training partition only, then transform its validation partition.

# Fit the full pipeline after evaluation
model.fit(X, y)

# Three new raw rows. No manual imputation, scaling, or encoding is needed.
new_rows = pd.DataFrame([
    {
        "mean radius": 14.0,
        "mean texture": 20.0,
        "mean perimeter": 90.0,
        "mean area": 600.0,
        "mean smoothness": 0.10,
        "mean compactness": 0.12,
        "mean concavity": 0.10,
        "mean concave points": 0.06,
        "mean symmetry": 0.18,
        "mean fractal dimension": 0.06,
        "radius error": 0.30,
        "texture error": 1.0,
        "perimeter error": 2.0,
        "area error": 25.0,
        "smoothness error": 0.007,
        "compactness error": 0.03,
        "concavity error": 0.03,
        "concave points error": 0.01,
        "symmetry error": 0.02,
        "fractal dimension error": 0.004,
        "worst radius": 17.0,
        "worst texture": 25.0,
        "worst perimeter": 110.0,
        "worst area": 850.0,
        "worst smoothness": 0.14,
        "worst compactness": 0.25,
        "worst concavity": 0.25,
        "worst concave points": 0.10,
        "worst symmetry": 0.30,
        "worst fractal dimension": 0.08,
        "region": "North",
    },
    {
        "mean radius": 20.0,
        "mean texture": 25.0,
        "mean perimeter": 130.0,
        "mean area": 1200.0,
        "mean smoothness": 0.11,
        "mean compactness": 0.20,
        "mean concavity": 0.25,
        "mean concave points": 0.12,
        "mean symmetry": 0.20,
        "mean fractal dimension": 0.07,
        "radius error": 0.50,
        "texture error": 1.2,
        "perimeter error": 3.0,
        "area error": 60.0,
        "smoothness error": 0.008,
        "compactness error": 0.04,
        "concavity error": 0.05,
        "concave points error": 0.02,
        "symmetry error": 0.03,
        "fractal dimension error": 0.005,
        "worst radius": 25.0,
        "worst texture": 35.0,
        "worst perimeter": 170.0,
        "worst area": 2000.0,
        "worst smoothness": 0.16,
        "worst compactness": 0.40,
        "worst concavity": 0.50,
        "worst concave points": 0.20,
        "worst symmetry": 0.35,
        "worst fractal dimension": 0.10,
        "region": "South",
    },
    {
        "mean radius": 12.0,
        "mean texture": 18.0,
        "mean perimeter": 78.0,
        "mean area": 450.0,
        "mean smoothness": 0.09,
        "mean compactness": 0.08,
        "mean concavity": 0.04,
        "mean concave points": 0.03,
        "mean symmetry": 0.17,
        "mean fractal dimension": 0.06,
        "radius error": 0.20,
        "texture error": 0.8,
        "perimeter error": 1.5,
        "area error": 15.0,
        "smoothness error": 0.006,
        "compactness error": 0.02,
        "concavity error": 0.02,
        "concave points error": 0.008,
        "symmetry error": 0.02,
        "fractal dimension error": 0.003,
        "worst radius": 14.0,
        "worst texture": 22.0,
        "worst perimeter": 90.0,
        "worst area": 600.0,
        "worst smoothness": 0.12,
        "worst compactness": 0.15,
        "worst concavity": 0.10,
        "worst concave points": 0.06,
        "worst symmetry": 0.25,
        "worst fractal dimension": 0.07,
        "region": "Central",
    },
])

predictions = model.predict(new_rows)
print("Predictions for new rows:", predictions.tolist())

# Save and reload the complete fitted pipeline
joblib.dump(model, "classification_pipeline.joblib")
loaded_model = joblib.load("classification_pipeline.joblib")

print("Predictions after loading:", loaded_model.predict(new_rows).tolist())