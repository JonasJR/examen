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

iris = load_iris()
data = iris.data
target = iris.target
c = 0
for i in target:
    if i == 0:
        c += 1
        print(str(c))
        data = np.delete(data, np.where(target==i), 0)
        target = np.delete(target, np.where(target==i), 0)
#We create the algorithm
#svc = linear_model.LinearRegression()
#svc = SVC(kernel='linear', C=1.0)
svc = tree.DecisionTreeClassifier()
#svc = neighbors.KNeighborsClassifier()

f = open('irisno0-2.txt', 'w')
f.write("Algoritm: DecisionTreeClassifier\n\n")

scores = []
size = []
loop = [0.95,0.9]
#We create a loop for the process
for i in loop:
    #We split the data into train and test data.
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=i, random_state=32)
    svc.fit(data_train,target_train)
    pred = svc.predict(data_test)
    true = target_test
    conf_mat = confusion_matrix(true,pred,labels=[1,2])
    #size.append(str(len(target_train)))
    sym = 0
    osym = 0
    dubsym = 0
    for i in target_train:
        if i == 1:
            sym += 1
        elif i == 2:
            dubsym += 1
        else:
            osym += 1
    scores.append(svc.score(data_test,target_test))
    f.write("Train size: " + str(len(target_train)) + "\n")
    f.write("Target 0 in training: " + str(osym) + "\nTarget 1 in training: " + str(sym) + " \nTarget 2 in training: " + str(dubsym) + "\n")
    f.write("Test size: " + str(len(target_test)) + "\n")
    f.write("Score: " + str(svc.score(data_test,target_test)) + "\n")
    f.write("Confusion Matrix: \n")
    f.write(str(conf_mat))
    f.write("\n\n\n\n")

f.write("-------------------------------------------\n\n")

svc = neighbors.KNeighborsClassifier()
f.write("Algoritm: KNN\n\n")
scores = []
size = []
loop = [0.95,0.9]
#We create a loop for the process
for i in loop:
    #We split the data into train and test data.
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=i, random_state=32)
    svc.fit(data_train,target_train)
    pred = svc.predict(data_test)
    true = target_test
    conf_mat = confusion_matrix(true,pred,labels=[1,2])
    #size.append(str(len(target_train)))
    sym = 0
    osym = 0
    dubsym = 0
    for i in target_train:
        if i == 1:
            sym += 1
        elif i == 2:
            dubsym += 1
        else:
            osym += 1
    scores.append(svc.score(data_test,target_test))
    f.write("Train size: " + str(len(target_train)) + "\n")
    f.write("Target 0 in training: " + str(osym) + "\nTarget 1 in training: " + str(sym) + " \nTarget 2 in training: " + str(dubsym) + "\n")
    f.write("Test size: " + str(len(target_test)) + "\n")
    f.write("Score: " + str(svc.score(data_test,target_test)) + "\n")
    f.write("Confusion Matrix: \n")
    f.write(str(conf_mat))
    f.write("\n\n\n\n")

f.write("-------------------------------------------\n\n")

svc = SVC(kernel='linear', C=1.0)
f.write("Algoritm: SVC\n\n")
scores = []
size = []
loop = [0.95,0.9]
#We create a loop for the process
for i in loop:
    #We split the data into train and test data.
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=i, random_state=32)
    svc.fit(data_train,target_train)
    pred = svc.predict(data_test)
    true = target_test
    conf_mat = confusion_matrix(true,pred,labels=[1,2])
    #size.append(str(len(target_train)))
    sym = 0
    osym = 0
    dubsym = 0
    for i in target_train:
        if i == 1:
            sym += 1
        elif i == 2:
            dubsym += 1
        else:
            osym += 1
    scores.append(svc.score(data_test,target_test))
    f.write("Train size: " + str(len(target_train)) + "\n")
    f.write("Target 0 in training: " + str(osym) + "\nTarget 1 in training: " + str(sym) + " \nTarget 2 in training: " + str(dubsym) + "\n")
    f.write("Test size: " + str(len(target_test)) + "\n")
    f.write("Score: " + str(svc.score(data_test,target_test)) + "\n")
    f.write("Confusion Matrix: \n")
    f.write(str(conf_mat))
    f.write("\n\n\n\n")
f.close()
