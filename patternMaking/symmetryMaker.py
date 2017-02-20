import csv
import numpy as np
import itertools

data = []

seq = itertools.product("01", repeat=16)

for s in seq:
    arr = np.fromiter(s, np.int8).reshape(4, 4)
    data.append(arr)

with open("fulldata.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(data)
