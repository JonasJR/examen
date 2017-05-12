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
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import KFold


#Two lines to ignore an error message about falling back to a gles driver
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
#Print out full arrays
np.set_printoptions(threshold=np.nan)

data = np.loadtxt("trainingdata.csv", delimiter=',', usecols=(range(0,63)))
target = np.loadtxt("trainingdata.csv", delimiter=',', usecols=(64))

algorithms = [neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform'), tree.DecisionTreeClassifier(criterion='gini'), SVC(kernel='linear', C=1.0)]

#We create the algorithm
#alg = SVC(kernel='linear', C=1.0)
#alg = tree.DecisionTreeClassifier(criterion='gini')
#alg = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')

f = open('TEEEEST.txt', 'w')

for alg in algorithms:
    f.write("Algoritm: " + str(alg) + "\n\n")
    scores = []
    size = []
    loop = [40,4]
    #We create a loop for the process
    for i in loop:
        #We split the data into train and test data.
        cv = KFold(n=len(target),n_folds=i)
        for j, (train, test) in enumerate(cv):
            alg.fit(data[train],target[train])
            pred = alg.predict(data[test])
            true = target[test]
            conf_mat = confusion_matrix(true,pred,labels=[0,1])
            scores.append(alg.score(data[test],target[test]))
            f.write("Train size: " + str(len(target[train])) + "\n")
            f.write("Test size: " + str(len(target[test])) + "\n")
            f.write("Score: " + str(alg.score(data[test],target[test])) + "\n")
            f.write("Confusion Matrix: \n")
            f.write(str(conf_mat))
            f.write("\n\n\n\n")
f.close()
