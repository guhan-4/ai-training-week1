from fastapi import FastAPI

app = FastAPI()

students = [
    {"id": 1, "name": "Guhan","city":"Thanjavur","email":"cantsay@gmail.com","salary":"80k"},
    {"id": 2, "name": "Rajan","city":"Thanjavur","email":"cantsay@gmail.com","salary":"80k"},
]

@app.get("/")
def home():
    return {"mess": "Hello world...."}

@app.post("/addstudent")
def addstudent(student: dict):
    students.append(student)
    return {
        "message": "Student added",
        "student": student
    }

@app.get("/student")
def get_students():
    return students