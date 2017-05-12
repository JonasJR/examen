from sklearn.datasets import load_digits, load_iris
from sklearn.svm import SVC
from sklearn import tree
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#Two lines to ignore an error message about falling back to a gles driver
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
#Print out full arrays
np.set_printoptions(threshold=np.nan)

data = np.loadtxt("trainingdata.csv", delimiter=',', usecols=(range(0,63)))
target = np.loadtxt("trainingdata.csv", delimiter=',', usecols=(64))

print("------")
print("Size of total dataset: " + str(len(target)))

sym = 0
non = 0
for i in target:
    if i == 1:
        sym += 1
    elif i == 0:
        non += 1

print("Sym in dataset: " + str(sym))
print("Not sym in dataset: " + str(non))
loop = [0.995,0.99,0.985,0.98,0.97,0.95,0.9,0.8,0.7,0.6,0.5]
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=loop[0], random_state=32)

trainSym = 0
trainNon = 0
for i in target_train:
    if i == 1:
        trainSym += 1
    elif i == 0:
        trainNon += 1
print("-----")

print("Total training size: " + str(len(target_train)))
print("Sym in TRAINING: " + str(trainSym))
print("Non sym in TRAINING: " + str(trainNon))

print("-----")

testSym = 0
testNon = 0
for i in target_test:
    if i == 1:
        testSym += 1
    elif i == 0:
        testNon += 1
print("-----")
print("Total test size: " + str(len(target_test)))
print("Sym in TESTING: " + str(testSym))
print("Non sym in TESTING: " + str(testNon))
