import numpy as np
marks = np.array(
[

    [70, 80, 90],
    [60, 75, 85],
    [88, 92, 79],
    [55, 65, 70]

])
#4x3 4 rows as students 3 colums as subjects

print(marks)
print(marks.shape)
print(marks.dtype)
print(marks.size)

print(marks[:2,-1]) 

# print(marks[:2,-2])

print(marks+5) 

student_avg=np.round(marks.mean(axis=1),2)
subject_avg=np.round(marks.mean(axis=0),2)

print("STUDENT AVG:",student_avg) #axis 1 = row
print("SUBJECT AVG:",subject_avg) #axis 0 = column

weights=[0.3,0.3,0.4]

print("WEIGHTED SCORES:",np.dot(subject_avg,weights))
print("WEIGHTED SCORES:",np.dot(marks,weights))


val=np.arange(12)
print(val.reshape(3,4))