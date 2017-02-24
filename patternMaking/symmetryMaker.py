import csv
import numpy as np
import itertools

data = np.array([[],[],[],[]])
test = np.array([[1,2,3,4],[5,6,7,8]])
seq = itertools.product("01", repeat=16)
counter = 0
for s in seq:
    arr = np.fromiter(s, np.int8).reshape(4, 4)
    print(arr)
    data = np.append(data, arr)
    #data.append(arr)
    if counter == 2:
        break
    counter += 1
print("TEST")
print(data)
d = np.asarray(data)
np.savetxt("test.csv", data, delimiter=",")
#with open("fulldata.csv", "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(data)
