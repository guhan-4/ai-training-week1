import pandas as pd
import matplotlib.pyplot as plt
chart = pd.read_csv("C:/Users/Lenovo/ai-training-week1/Day4_Task/employee.csv")
team_avg_salary = chart.groupby("department")["salary"].mean()


bars = plt.bar(team_avg_salary.index, team_avg_salary.values)
plt.title("Average Salary by Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")
plt.bar_label(bars, fmt="%.2f", padding=3)
plt.tight_layout()
plt.savefig("Bar_chart.png")
plt.show()

plt.hist(chart["salary"], bins=5, edgecolor="black")
plt.title("Distribution of Employee Salaries")
plt.xlabel("Salary")
plt.ylabel("Number of Employees")
plt.savefig("Histogram_chart.png")
plt.tight_layout()
plt.show()



for department, group in chart.groupby("department"):
    plt.scatter(group["id"], group["salary"], label=department)

plt.title("Employee ID vs Salary by Department")
plt.xlabel("Employee ID")
plt.ylabel("Salary")
plt.legend(title="Department")
plt.tight_layout()
plt.savefig("Scatter_plot.png")
plt.show()



months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [12, 15, 14, 18, 22, 25]

plt.plot(months, sales, marker="o")
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("Line_chart_monthly_sales.png")
plt.show()
plt.close()