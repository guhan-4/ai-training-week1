import pandas as pd

order = pd.read_csv("C:/Users/Lenovo/Desktop/Python/TASK/orders.csv")
print(order)
print(order.isna().sum())

# print(order.isna())
print("dup\n",order.duplicated())

# Remove duplicate rows............................................................
order = order.drop_duplicates()

# Make amount numeric and fill missing values with its median
order["amount"] = pd.to_numeric(order["amount"], errors="coerce")
order["amount"] = order["amount"].fillna(order["amount"].median())

# Make order_id a string
order["order_id"] = order["order_id"].astype(str)

# Aggregate by customer
result = order.groupby("customer_id").agg(
    total=("amount", "sum"),
    orders=("order_id", "count")
)

print("\nTHE RESULT IS:\n",result)