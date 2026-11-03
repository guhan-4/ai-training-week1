import pandas as pd
df=pd.read_csv("C:/Users/Lenovo/Desktop/Python/TASK/Day5_Task/Day5_FH_Task2/Emp_data.csv")
import pandas as pd

df["id"] = pd.to_numeric(df["id"], errors="coerce")
df["salary"] = pd.to_numeric(df["salary"], errors="coerce")

print("\n1. Summary statistics")
print(df[["id", "salary"]].agg(
    ["mean", "median", "min", "max", "std"]
).T.round(2))

print("\n2. Summary by department")
print(df.groupby("department").agg(
    employee_count=("id", "count"),
    average_salary=("salary", "mean"),
    median_salary=("salary", "median"),
    total_salary=("salary", "sum")
).round(2))

print("\nSummary by city")
print(df.groupby("city").agg(
    employee_count=("id", "count"),
    average_salary=("salary", "mean"),
    median_salary=("salary", "median"),
    total_salary=("salary", "sum")
).round(2))

# Correlation between numeric columns........................................
print("\n3. Correlation matrix")
print(df[["id", "salary"]].corr().round(3))
print(
    "The only distinct correlation pair is id and salary: "
    f"{df['id'].corr(df['salary']):.3f}"
)

# Business questions with numeric answers.................................
print("\n4. Business questions and answers")

print(f"Q1. What is the average salary? Answer: {df['salary'].mean():,.2f}")

dept_avg_salary = df.groupby("department")["salary"].mean()
print(
    "Q2. Which department has the highest average salary? "
    f"Answer: {dept_avg_salary.idxmax()} "
    f"({dept_avg_salary.max():,.2f})"
)

city_counts = df["city"].value_counts()
print(
    "Q3. Which city has the most employees? "
    f"Answer: {city_counts.idxmax()} ({city_counts.max()} employees)"
)

print(f"Q4. What is the median salary? Answer: {df['salary'].median():,.2f}")

dept_total_salary = df.groupby("department")["salary"].sum()
print(
    "Q5. Which department has the highest total salary? "
    f"Answer: {dept_total_salary.idxmax()} "
    f"({dept_total_salary.max():,.2f})"
)

department_budget = pd.DataFrame({
    "department": ["Sales", "HR", "IT", "Finance"],
    "budget": [100000, 120000, 250000, 150000]
})

merged_df = pd.merge(
    df,
    department_budget,
    on="department",
    how="left",
    validate="many_to_one"
)

print("\n5. Employee data merged with department budgets")
print(merged_df)