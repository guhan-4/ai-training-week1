import pandas as pd

# Load both CSV files
order = pd.read_csv("C:/Users/Lenovo/Desktop/Python/TASK/orders.csv")
customers = pd.read_csv("C:/Users/Lenovo/Desktop/Python/TASK/customers.csv")

# Check missing values and duplicates
print("Missing values:\n", order.isna().sum())
print("Duplicate rows:\n", order.duplicated())

# Clean orders
order = order.drop_duplicates()
order["amount"] = pd.to_numeric(order["amount"], errors="coerce")
order["amount"] = order["amount"].fillna(order["amount"].median())
order["order_id"] = order["order_id"].astype(str)

# Aggregate orders by customer
result = order.groupby("customer_id").agg(
    total=("amount", "sum"),
    orders=("order_id", "count")
)
print("\nCustomer totals:\n", result)

# Joins using the cleaned order data
inner_join = order.merge(customers, on="customer_id", how="inner")
print("\nInner join:\n", inner_join)

left_join = customers.merge(order, on="customer_id", how="left")
print("\nLeft join:\n", left_join)

print("\nCustomers with no orders:\n",
      left_join.loc[left_join["order_id"].isna(), ["customer_id", "name", "city"]])

outer_join = order.merge(
    customers, on="customer_id", how="outer", indicator=True
)
print("\nOuter join:\n", outer_join)

print("\nOrders with no customer record:\n",
      outer_join.loc[outer_join["_merge"] == "left_only"])