import csv
import numpy as np
import itertools
from sklearn import datasets
import random
import time

#Here we want to create all possible symmetric 4x4 images
#We start with creating a 3D array off all possible combinations in a 2x4 array.

data = ""
seq = itertools.product("01", repeat=9)
f = open('dubblesymandsym6x6.csv', 'w')
#Create a 2x4 array and make the symetry!
stringsarr = []
start = time.time()
counter = 0
for i in seq:
    arr = np.fromiter(i, np.int8).reshape(3, 3)
    strings = ""
    strings += "%s,%s,%s,%s,%s,%s," % (str(i[0]), str(i[1]), str(i[2]), str(i[2]), str(i[1]), str(i[0]))
    strings += "%s,%s,%s,%s,%s,%s," % (str(i[3]), str(i[4]), str(i[5]), str(i[5]), str(i[4]), str(i[3]))
    strings += "%s,%s,%s,%s,%s,%s," % (str(i[6]), str(i[7]), str(i[8]), str(i[8]), str(i[7]), str(i[6]))
    strings += "%s,%s,%s,%s,%s,%s," % (str(i[6]), str(i[7]), str(i[8]), str(i[8]), str(i[7]), str(i[6]))
    strings += "%s,%s,%s,%s,%s,%s," % (str(i[3]), str(i[4]), str(i[5]), str(i[5]), str(i[4]), str(i[3]))
    strings += "%s,%s,%s,%s,%s,%s," % (str(i[0]), str(i[1]), str(i[2]), str(i[2]), str(i[1]), str(i[0]))
    strings += "2"
    stringsarr.append(strings)
    #print("1: "+str(time.time() - start))
    counter += 1
    if counter == 512:
        break
random.shuffle(stringsarr)
for i in stringsarr:
    f.write(i + "\n")


data = ""
seq = itertools.product("01", repeat=18)
#Create a 2x4 array and make the symetry!
start = time.time()
counter = 0
for i in seq:
    arr = np.fromiter(i, np.int8).reshape(6, 3)
    strings = ""
    for j in arr:
        one = random.getrandbits(1)
        two = random.getrandbits(1)
        three = random.getrandbits(1)
        temp = "%s,%s,%s,%s,%s,%s," % (str(one), str(two), str(three), str(three), str(two), str(one))
        strings += temp
        temp = ""
    strings += "1"
    f.write(strings + "\n")
    #print("1: "+str(time.time() - start))
    counter += 1
    if counter == 512:
        break

print("0 started!!!!!")
seq2 = itertools.product("01", repeat=18)
#Just do the same as last time but with non symmetric
c = 0
for i in seq2:
    arr = np.fromiter(i, np.int8).reshape(6, 3)
    strings = ""
    for j in arr:
        one = random.getrandbits(1)
        two = random.getrandbits(1)
        three = random.getrandbits(1)
        four = random.getrandbits(1)
        five = random.getrandbits(1)
        six = random.getrandbits(1)
        temp = "%s,%s,%s,%s,%s,%s," % (str(one), str(two), str(three), str(four), str(five), str(six))
        strings += temp
        temp = ""
    strings += "0"
    f.write(strings + "\n")
    #print("0: "+str(time.time() - start))
    c += 1
    if c == 512:
        break
f.close()
