import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
#Two lines to ignore an error message about falling back to a gles driver
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

#According to the scikit webpage http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
#we should be using the SVC to predict the symmetry since we are predicting a category
#and we have labeled data and we have labeled data and less than 100K samples.

#So lets load the SVC
svc = svm.SVC()

#Then we load the trainingdata from our created file
data = np.loadtxt("trainingdata.csv", delimiter=',', usecols=(0,1,2,3))
target = np.loadtxt("trainingdata.csv", delimiter=',', usecols=(4))

#Then we fit (train) the svc with the trainingdata
svc.fit(data, target)

#Lets seee what happens when we predict a new array that we create
test = np.loadtxt("testingdata.csv", delimiter=',', skiprows=1, usecols=(0,1,2,3))
pred = svc.predict(test)
print(pred)

#Well, lets try another one!
linreg = linear_model.LinearRegression()
linreg.fit(data, target)
pred2 = linreg.predict(test)
#print(pred2)

#And another one
sgd = linear_model.SGDClassifier()
sgd.fit(data, target)
pred3 = sgd.predict(test)
#print(pred3)

#LogisticRegression
logistic = linear_model.LogisticRegression()
logistic.fit(data, target)
pred4 = logistic.predict(test)
#print(pred4)

#And a last one!
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(data, target)
pred5 = knn.predict(test)
#print(pred5)
