import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import datasets
#Two lines to ignore an error message about falling back to a gles driver
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

#Load the Iris dataset. This loads ALL the 150 data into the variable iris.
iris = datasets.load_iris()

#We devide the dataset into a data and target
#where data is sepal length and width etc. and target is family name
data = iris.data
target = iris.target

#Make a curve about the error rate
size = 0.50
#We devide it into a training part and a testing part
train_data, test_data, train_target, test_target = train_test_split(data, target, random_state=42, test_size=size)

#Loads an SVC (Support Vector Classification) to be used in the learning and classification
svc = svm.SVC()

#Loads an LinearRegression to be used in the learning and classification for comparing
linreg = linear_model.LinearRegression()

#Here we place the training data and training target into the svc and linreg. This is where
#the training is done. After this we can start predicting.
svc.fit(train_data, train_target)
linreg.fit(train_data, train_target)

#Here we predict the outcome of using the test_data as input for both SVC and LinReg
svc_target_pred = svc.predict(test_data)
linreg_target_pred = linreg.predict(test_data)

#Prints out the results to the screen
#print("When using SVC: \n" + str(svc_target_pred) + "\nAnd the true target: \n" + str(test_target))#"\n\nWhen using LinearRegression: \n" + str(linreg_target_pred))



#Now we try to predict some crazy input! So we create some crazy data
crazy_data = np.array([[-10.0,165.13,1.5,-20.2]])

#We make prediction with the crazy data
svc_crazy_pred = svc.predict(crazy_data)
linreg_crazy_pred = linreg.predict(crazy_data)
#print("When using SVC: \n" + str(svc_crazy_pred) + "\n\nWhen using LinearRegression: \n" + str(linreg_crazy_pred))

#Easy to see the error in the Linear Regression, since it returns a -5.32... value.
#But impossible to see the fault in the SVC.





#Lets try some other algorithms!!!
#According to the Sci-Kit website: http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
#we should be using the LinearSVC but lets try what happens with other algorithms!

#Lets try the SGD (Stochastic Gradient Descent)
sgd = linear_model.SGDClassifier()
sgd.fit(train_data, train_target)
sgd_target_pred = sgd.predict(test_data)
sgd_crazy_pred = sgd.predict(crazy_data)

print("Prediction data: \n" + str(sgd_target_pred) + "\n\True data: \n" + str(test_target))

#Conclusion:
#The scikit site have a pretty good knowledge of what algorithms to use, so lets follow
#the chart to know what algorithm to use for the symmetry learning.
