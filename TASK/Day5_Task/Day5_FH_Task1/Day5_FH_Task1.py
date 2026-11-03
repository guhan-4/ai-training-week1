import pandas as pd
df = pd.read_csv("C:/Users/Lenovo/Desktop/Python/TASK/Day5_Task/Day5_FH_Task1/customer_data.csv")

before = {
    "Stage": "Before",
    "Rows": len(df),
    "MissingValues": int(df.isna().sum().sum()),
    "Duplicates": int(df.duplicated().sum()),
}

df["name"] = df["name"].str.strip().str.title()
df["city"] = df["city"].str.strip().str.title()
df["city"]=df["city"].fillna("unknown")

df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
df["age"] = pd.to_numeric(df["age"], errors="coerce")

df = df.drop_duplicates()

df["age"] = df["age"].fillna(df["age"].median())

q1 = df["age"].quantile(0.25)
q3 = df["age"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df["age"] = df["age"].clip(lower=lower_bound, upper=upper_bound)
df["age"] = df["age"].round().astype(int)

df.to_csv("clean_data.csv", index=False)

after = {
    "Stage": "After",
    "Rows": len(df),
    "MissingValues": int(df.isna().sum().sum()),
    "Duplicates": int(df.duplicated().sum()),
}

pd.DataFrame([before, after]).to_csv("before_after_summary.csv", index=False)

print(pd.DataFrame([before, after]).to_string(index=False))
print("\nSaved clean_data.csv and before_after_summary.csv")