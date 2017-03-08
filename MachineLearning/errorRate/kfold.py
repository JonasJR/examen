from sklearn.datasets import load_digits, load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import numpy as np
import math
import random

#We start of by loading the iris datasets and store them in data and target
iris = load_digits()
data, target = iris.data, iris.target

#We create the cross validator
svc = SVC(kernel='linear', C=1.0)

#We create an empty copy of the dataset and use numpy (np) to make sure they are the same size
shuffle = np.arange(len(data))
#we then shuffle the indexes of the dataset using numpy
np.random.shuffle(shuffle)
#we then store the shuffled data in data and the shuffled target in target
#this way we use the same indexing and make sure that the correct target and data is asosiated
data = data[shuffle]
target = target[shuffle]

#We split the data into "size" test and the rest as learning
size = 0.998
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=size, random_state=42)
#We fit and score
svc.fit(X_train,y_train)
print("\nScore: " + str(svc.score(X_test,y_test)))
print("\nPred Target: " + str(svc.predict(X_test)))
print("\nTest Target: " + str(y_test))
print("\nTraining: " + str(y_train))
