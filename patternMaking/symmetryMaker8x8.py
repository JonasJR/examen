import csv
import numpy as np
import itertools
from sklearn import datasets
import random
import time

#Here we want to create all possible symmetric 4x4 images
#We start with creating a 3D array off all possible combinations in a 2x4 array.

data = ""
seq = itertools.product("01", repeat=16)
f = open('dubblesym8x8.csv', 'w')
#Create a 2x4 array and make the symetry!
stringsarr = []
start = time.time()
counter = 0
for i in seq:
    arr = np.fromiter(i, np.int8).reshape(4, 4)
    strings = ""
    one = random.getrandbits(1)
    two = random.getrandbits(1)
    three = random.getrandbits(1)
    four = random.getrandbits(1)
    five = random.getrandbits(1)
    six = random.getrandbits(1)
    seven = random.getrandbits(1)
    eight = random.getrandbits(1)
    nine = random.getrandbits(1)
    ten = random.getrandbits(1)
    eleven = random.getrandbits(1)
    twelve = random.getrandbits(1)
    thirteen = random.getrandbits(1)
    fourteen = random.getrandbits(1)
    fiftheen = random.getrandbits(1)
    sixteen = random.getrandbits(1)
    strings += "%s,%s,%s,%s,%s,%s,%s,%s," % (one, two, three, four, four, three, two, one)
    strings += "%s,%s,%s,%s,%s,%s,%s,%s," % (five, six, seven, eight, eight, seven, six, five)
    strings += "%s,%s,%s,%s,%s,%s,%s,%s," % (nine, ten, eleven, twelve, twelve, eleven, ten, nine)
    strings += "%s,%s,%s,%s,%s,%s,%s,%s," % (thirteen, fourteen, fiftheen, sixteen, sixteen, fiftheen, fourteen, thirteen)
    strings += "%s,%s,%s,%s,%s,%s,%s,%s," % (thirteen, fourteen, fiftheen, sixteen, sixteen, fiftheen, fourteen, thirteen)
    strings += "%s,%s,%s,%s,%s,%s,%s,%s," % (nine, ten, eleven, twelve, twelve, eleven, ten, nine)
    strings += "%s,%s,%s,%s,%s,%s,%s,%s," % (five, six, seven, eight, eight, seven, six, five)
    strings += "%s,%s,%s,%s,%s,%s,%s,%s," % (one, two, three, four, four, three, two, one)
    strings += "1"
    stringsarr.append(strings)
    #print("1: "+str(time.time() - start))
    counter += 1
    if counter == 1000:
        break
random.shuffle(stringsarr)
for i in stringsarr:
    f.write(i + "\n")

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
    #print("0: "+str(time.time() - start))
    c += 1
    if c == 1000:
        break
f.close()
