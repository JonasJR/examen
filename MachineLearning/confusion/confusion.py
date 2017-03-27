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

#We start of by loading the digits datasets and store them in data and target, Then we add 3 items to make it 1800 items
digits = load_digits()
data, target = digits.data, digits.target
data = np.append(data,[data[0],data[1],data[2]],axis=0)
target = np.append(target,[target[0],target[1],target[2]],axis=0)

#We create the algorithm
#svc = linear_model.LinearRegression()
#svc = SVC(kernel='linear', C=1.0)
svc = tree.DecisionTreeClassifier()
#svc = neighbors.KNeighborsClassifier()

f = open('outputTree.txt', 'w')
f.write("Algoritm: Tree\n\n")

scores = []
size = []
loop = [0.995,0.99,0.985,0.98,0.97,0.95,0.9,0.8,0.7,0.6,0.5]
#We create a loop for the process
for i in loop:
    #We split the data into train and test data.
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=i, random_state=32)
    svc.fit(data_train,target_train)
    pred = svc.predict(data_test)
    true = target_test
    conf_mat = confusion_matrix(true,pred,labels=[0,1,2,3,4,5,6,7,8,9])
    #size.append(str(len(target_train)))
    scores.append(svc.score(data_test,target_test))
    f.write("Train size: " + str(len(target_train)) + "\n")
    f.write("Test size: " + str(len(target_test)) + "\n")
    f.write("Score: " + str(svc.score(data_test,target_test)) + "\n")
    f.write("Confusion Matrix: \n")
    f.write(str(conf_mat))
    f.write("\n\n\n\n")
f.close()
