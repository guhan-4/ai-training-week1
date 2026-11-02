from fastapi import FastAPI
app=FastAPI()

students=[
    {
        "id":1,
        "name":"Guhan"
    },
    {
        "id":2,
        "name":"Rajan"

    }
]

@app.get("/")
def home():
    return{
        "message":"Hello world...."
    }