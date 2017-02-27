import csv
import numpy as np
import itertools

#Here we want to create all possible symmetric 4x4 images
#We start with creating a 3D array off all possible combinations in a 2x4 array.

data = []
seq = itertools.product("01", repeat=8)
counter = 0 # remove after testing is done
for s in seq:
    arr = np.fromiter(s, np.int8).reshape(4, 2)
    nparr = np.array(arr)
    #data = np.append(data, arr)
    data.append(nparr)
    print("EN ARRAY:")
    print(nparr)

    #just for testing:
    if counter == 2:
        break
    counter += 1
    #end of testing

d = np.asarray(data)
print("TEST")
print(d)
np.savetxt("test.csv", d, delimiter=",")
#with open("fulldata.csv", "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(data)
