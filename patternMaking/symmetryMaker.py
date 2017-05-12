import csv
import numpy as np
import itertools
from sklearn import datasets
import random
import time

#Here we want to create all possible symmetric 4x4 images
#We start with creating a 3D array off all possible combinations in a 2x4 array.

data = ""
seq = itertools.product("01", repeat=32)
f = open('trainingdata.csv', 'w')
#Create a 2x4 array and make the symetry!
counter = 0
start = time.time()
for s in seq:
    counter += 1
    if counter == 1000000:
        arr = np.fromiter(s, np.int8).reshape(8, 4)
        strings = ""
        #loop through the array
        for i in arr:
            #change them to strings
            temp = "%s,%s,%s,%s,%s,%s,%s,%s," % (str(i[0]),str(i[1]),str(i[2]),str(i[3]),str(i[3])[::-1],str(i[2])[::-1],str(i[1])[::-1],str(i[0])[::-1])
            #add them to one string with , at end
            strings += temp
            temp = ""
        #adds a 1 for target training and make sure last one does not have the ending ,
        strings += "1"
        #add them to one string
        f.write(strings + "\n")
        counter = 0
        print("1: "+str(time.time() - start))

seq2 = itertools.product("01", repeat=32)
#Just do the same as last time but with non symmetric
c = 0
for s in seq2:
    c += 1
    if c == 1000000:
        arr = np.fromiter(s, np.int8).reshape(8, 4)
        strings = ""
        #loop through the array
        for i in arr:
            #This time just add random bits
            temp =  "%s,%s,%s,%s,%s,%s,%s,%s," % (str(random.getrandbits(1)),str(random.getrandbits(1)),str(random.getrandbits(1)),str(random.getrandbits(1)),str(random.getrandbits(1)),str(random.getrandbits(1)),str(random.getrandbits(1)),str(random.getrandbits(1)))
            #add them to one string with , at end
            strings += temp
            temp = ""
        #adds a 0 for target training and make sure last one does not have the ending ,
        strings += "0"
        #add them to one string
        f.write(strings + "\n")
        c = 0
        print("0: "+str(time.time() - start))
f.close()
