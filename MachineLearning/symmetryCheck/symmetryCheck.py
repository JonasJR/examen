import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm

#According to the scikit webpage http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
#we should be using the SVC to predict the symmetry since we are predicting a category
#and we have labeled data and we have labeled data and less than 100K samples.

#So lets load the SVC
svc = svm.SVC()

#Then we load the trainingdata from our created file
data = np.loadtxt("trainingdata.csv", delimiter=',', skiprows=1, usecols=(0,1,2,3))
target = np.loadtxt("trainingdata.csv", delimiter=',', skiprows=1, usecols=(4))

#Then we fit (train) the svc with the trainingdata
svc.fit(data, target)

#Lets seee what happens when we predict a new array that we create
test = np.loadtxt("testingdata.csv", delimiter=',', skiprows=1, usecols=(0,1,2,3))
pred = svc.predict(test)
print(pred)
