from sklearn.datasets import load_digits, load_iris
from sklearn.svm import SVC
from sklearn import tree
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
#Two lines to ignore an error message about falling back to a gles driver
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

np.set_printoptions(threshold=np.nan)

#We start of by loading the iris datasets and store them in data and target
iris = load_digits()
data, target = iris.data, iris.target
data = np.append(data,[data[0],data[1],data[2]],axis=0)
target = np.append(target,[target[0],target[1],target[2]],axis=0)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.6, random_state=32)
tempX_train, tempX_test, tempy_train, tempy_test = train_test_split(X_test,y_test,test_size=0.9,random_state=32)
#We create the cross validator
#svc = linear_model.LinearRegression()
svc = SVC(kernel='linear', C=1.0)
#svc = tree.DecisionTreeClassifier()
#svc = neighbors.KNeighborsClassifier()

#We define a function for creating the randomized order of the dataset iris
#This function scrambles the data using numpy witch is a good tool for math
def randomize():
    #first we need to tell the function that data nad target are global
    global data, target
    #We create an empty copy of the dataset and use numpy (np) to make sure they are the same size
    shuffle = np.arange(len(data))
    #we then shuffle the indexes of the dataset using numpy
    np.random.shuffle(shuffle)
    #we then store the shuffled data in data and the shuffled target in target
    #this way we use the same indexing and make sure that the correct target and data is asosiated
    data = data[shuffle]
    target = target[shuffle]

#We define a function that loops through the kfold split array and gets the scores and create a median of it.
#v is the vector that was split by the kfold function
def main():
    #We start with calling the randomize function to make sure that the data is shuffled
    #print("Data: " + str(data[0]) + "\nTarget: " + str(target[0]))
    randomize()
    #print("Scrambled Data: " + str(data[0]) + "\nScrambled Target: " + str(target[0]))

    #We split the data into 3 chunks with 50 elements in each
    size = 900
    k = 2
    splitData = np.asarray(zip(*[iter(data)]*size))
    splitTarget = np.asarray(zip(*[iter(target)]*size))
    #print(len(splitTarget)) #Shows the number of groups (size 50 gives k = 3 and 3 equaly large groups)
    scores = []
    for i in range(0,k):
        tempData = splitData
        tempTarget = splitTarget
        testData = tempData[i]
        testTarget = tempTarget[i]
        np.delete(tempData,i,0)
        np.delete(tempTarget,i,0)
        learnData = []
        learnTarget = []
        for j in tempData:
            for k in j:
                learnData.append(k)
        for l in tempTarget:
            for p in l:
                learnTarget.append(p)
        svc.fit(np.asarray(learnData),np.asarray(learnTarget))
        #print(len(tempTarget))
        score = svc.score(np.asarray(testData), np.asarray(testTarget))
        scores.append(score)
    print(scores)
    #We run the score function on each group
    #score1 = svc.fit(np.asarray(splitData[1]+splitData[2]), np.asarray(splitTarget[1]+splitTarget[2])).score(np.asarray(splitData[0]), np.asarray(splitTarget[0]))
    #print(score1)
    #Then we do this on the two other groups
    #score2 = svc.fit(np.asarray(splitData[1]), np.asarray(splitTarget[1])).score(np.asarray(splitData[0]+splitData[2]), np.asarray(splitTarget[0]+splitTarget[2]))
    #print(score2)
    #score3 = svc.fit(np.asarray(splitData[2]), np.asarray(splitTarget[2])).score(np.asarray(splitData[1]+splitData[0]), np.asarray(splitTarget[1]+splitTarget[0]))
    #print(score3)
#main()
svc.fit(X_train,y_train)
#print(len(y_test))
print(len(y_train))
#print(y_train)
print("------")
#print(len(y_test))
#print(y_test)
print(svc.score(X_test,y_test))
