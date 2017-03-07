from sklearn.datasets import load_digits, load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import numpy as np

#We start of by loading the iris datasets and store them in data and target
dataset = load_iris()
data, target = dataset.data, dataset.target

scores = []
scores2 = []

#We create the cross validator and store the scores
svc = SVC(kernel='linear', C=1)
gnb = GaussianNB()

#We create a loop that creates scores of the clf with incrementing test size 1% to 99%
counter = 1.0
for i in range(0,98):
    size = counter/100 #For each iteration we increment the size with 0.1 witch represents 10%
    #We split the data into test and train with the given test_size
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=size, random_state=0)
    #We fit the svc and GaussianNB
    svc.fit(X_train, y_train)
    gnb.fit(X_train, y_train)
    scores.append(svc.score(X_test, y_test))
    scores2.append(gnb.score(X_test, y_test))
    #scores.append(cross_val_score(clf, data,target))
    counter +=1

#We create a plt to visualize the curves
#Create the figure
plt.figure()
#Set title
plt.title("")
#set x and y lables
plt.xlabel("Testing percentage")
plt.ylabel("Score")
#set it to grid style
plt.grid()
#set plot for svc and GaussianNB with coloring
plt.plot(scores, 'o-', label="SVC", color="r", linestyle="--")
plt.plot(scores2, 'o-', label="GaussianNB", color="g", linestyle="--")
#place the label in the top right
plt.legend(loc="best")
#show the figure
plt.show()

print("Data: " + str(dataset)) 

#print(cv)
