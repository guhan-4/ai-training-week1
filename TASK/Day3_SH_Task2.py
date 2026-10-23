import pandas as pd

customers = pd.read_csv("C:/Users/Lenovo/Desktop/Python/TASK/customers.csv")
order = pd.read_csv("C:/Users/Lenovo/Desktop/Python/TASK/orders.csv")
print(customers)

inner_join=order.merge(customers,on="customer_id",how="inner")
print("inner join:\n",inner_join)

left_join=customers.merge(order,on="customer_id",how="left")
print("left join:\n",left_join)

outer_join=order.merge(customers,on="customer_id",how="outer")
print("outer join:\n",outer_join)