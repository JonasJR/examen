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
f = open('trainingdata3.csv', 'w')
#Create a 2x4 array and make the symetry!
counter = 0
start = time.time()
counter = 0
for i in seq:
    arr = np.fromiter(i, np.int8).reshape(8, 4)
    strings = ""
    for j in arr:
        one = random.getrandbits(1)
        two = random.getrandbits(1)
        three = random.getrandbits(1)
        four = random.getrandbits(1)
        five = four
        six = three
        seven = two
        eight = one
        temp = "%s,%s,%s,%s,%s,%s,%s,%s," % (str(one), str(two), str(three), str(four), str(four), str(three), str(two), str(one))
        strings += temp
        temp = ""
    strings += "1"
    f.write(strings + "\n")
    print("1: "+str(time.time() - start))
    counter += 1
    if counter == 1000:
        break

print("0 started!!!!!")
seq2 = itertools.product("01", repeat=32)
#Just do the same as last time but with non symmetric
c = 0
for i in seq2:
    arr = np.fromiter(i, np.int8).reshape(8, 4)
    strings = ""
    for j in arr:
        one = random.getrandbits(1)
        two = random.getrandbits(1)
        three = random.getrandbits(1)
        four = random.getrandbits(1)
        five = random.getrandbits(1)
        six = random.getrandbits(1)
        seven = random.getrandbits(1)
        eight = random.getrandbits(1)
        temp = "%s,%s,%s,%s,%s,%s,%s,%s," % (str(one), str(two), str(three), str(four), str(five), str(six), str(seven), str(eight))
        strings += temp
        temp = ""
    strings += "0"
    f.write(strings + "\n")
    print("0: "+str(time.time() - start))
    c += 1
    if c == 1000:
        break
f.close()
