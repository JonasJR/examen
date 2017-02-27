import csv
import numpy as np
import itertools
from sklearn import datasets
import random

#Here we want to create all possible symmetric 4x4 images
#We start with creating a 3D array off all possible combinations in a 2x4 array.

data = ""
seq = itertools.product("01", repeat=8)

#Create a 2x4 array and make the symetry!
for s in seq:
    arr = np.fromiter(s, np.int8).reshape(4, 2)
    strings = ""
    #loop through the array
    for i in arr:
        #change them to strings
        temp = "" + str(i[0]) + "" + str(i[1]) + "" + str(i[1])[::-1] + "" + str(i[0])[::-1]
        #add them to one string with , at end
        strings += temp + ","
        temp = ""
    #adds a 1 for target training and make sure last one does not have the ending ,
    strings += "1"
    #add them to one string
    data += strings + "\n"

seq2 = itertools.product("01", repeat=8)
#Just do the same as last time but with non symmetric
for s in seq2:
    arr = np.fromiter(s, np.int8).reshape(4, 2)
    strings = ""
    #loop through the array
    for i in arr:
        #This time just add random bits
        temp = "" + str(random.getrandbits(1)) + "" + str(random.getrandbits(1)) + "" + str(random.getrandbits(1)) + "" + str(random.getrandbits(1))
        #add them to one string with , at end
        strings += temp + ","
        temp = ""
    #adds a 0 for target training and make sure last one does not have the ending ,
    strings += "0"
    #add them to one string
    data += strings + "\n"

#Save the string to a file
with open("trainingdata.csv", "wb") as f:
    f.write(data)
