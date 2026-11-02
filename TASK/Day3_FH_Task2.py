import pandas as pd
emp = pd.read_csv("C:/Users/Lenovo/Desktop/Python/TASK/employee.csv")
# print(emp)

# print("********************************************")

# print(emp.head())
# print(emp.describe())
# emp.info()

print(emp["name"])
print(emp[["name","salary"]])
print(emp.loc[emp["city"]=="Chennai","name"])
# print(emp.loc[emp["city"] == "Chennai", ["name", "salary"]])
print(emp.iloc[0:3,1:3])
print(emp.iloc[0:3,1])


hsal=emp[emp["salary"]>50000]
print(hsal)


high_salary = emp[emp["salary"] > 50000]
print(high_salary)
print("Salary above 50000 :", len(high_salary))

it_employees = emp[emp["department"] == "IT"]
print(it_employees)
print("IT employee :", len(it_employees)) 