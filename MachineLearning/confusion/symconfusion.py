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

data = np.loadtxt("trainingdata6x62.csv", delimiter=',', usecols=(range(0,35)))
target = np.loadtxt("trainingdata6x62.csv", delimiter=',', usecols=(36))

#We create the algorithm
#svc = linear_model.LinearRegression()
svc = SVC(kernel='linear', C=1.0)
#svc = tree.DecisionTreeClassifier()
#svc = neighbors.KNeighborsClassifier()

f = open('NEWoutputSymSVC6x62.txt', 'w')
f.write("Algoritm: SVC\n\n")

scores = []
size = []
loop = [0.975,0.75]
#We create a loop for the process
for i in loop:
    #We split the data into train and test data.
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=i, random_state=32)
    svc.fit(data_train,target_train)
    pred = svc.predict(data_test)
    true = target_test
    conf_mat = confusion_matrix(true,pred,labels=[0,1])
    #size.append(str(len(target_train)))
    scores.append(svc.score(data_test,target_test))
    f.write("Train size: " + str(len(target_train)) + "\n")
    f.write("Test size: " + str(len(target_test)) + "\n")
    f.write("Score: " + str(svc.score(data_test,target_test)) + "\n")
    f.write("Confusion Matrix: \n")
    f.write(str(conf_mat))
    f.write("\n\n\n\n")
f.close()
