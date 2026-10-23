# try:
#     a=10
#     b=0
#     c=a/b
#     print(c)

# except ZeroDivisionError:
#     print("can't divide by")


with open("../TEXT_FILES/hello.txt","r",encoding="utf-8")as f:
    result=f.read()

with open("../TEXT_FILES/hello.txt","w",encoding="utf-8")as f:
    result=f.write("day 2")

with open("../TEXT_FILES/hello.txt","a",encoding="utf-8")as f:
    result=f.write("/class")

with open("../TEXT_FILES/hello.txt","r",encoding="utf-8")as f:
    result=f.read()



print(result)