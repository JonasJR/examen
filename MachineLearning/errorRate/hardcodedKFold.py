from sklearn.datasets import load_digits, load_iris
from sklearn.svm import SVC
from sklearn import tree
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
import numpy as np
import math
import random
#Two lines to ignore an error message about falling back to a gles driver
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

#We start of by loading the iris datasets and store them in data and target
iris = load_iris()
data, target = iris.data, iris.target
#We create the cross validator
#svc = linear_model.LinearRegression()
svc = SVC(kernel='linear', C=1.0)
#svc = tree.DecisionTreeClassifier()

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
    print("Data: " + str(data[0]) + "\nTarget: " + str(target[0]))
    randomize()
    print("Scrambled Data: " + str(data[0]) + "\nScrambled Target: " + str(target[0]))

    #We split the data into 3 chunks with 50 elements in each
    size = 50
    k = 3
    splitData = np.asarray(zip(*[iter(data)]*size))
    splitTarget = np.asarray(zip(*[iter(target)]*size))
    print(len(splitTarget)) #Shows the number of groups (size 50 gives k = 3 and 3 equaly large groups)

    #We run the score function on each group
    score1 = svc.fit(np.asarray(splitData[0]), np.asarray(splitTarget[0])).score(np.asarray(splitData[1]+splitData[2]), np.asarray(splitTarget[1]+splitTarget[2]))
    print(score1)

    #Then we do this on the two other groups
    score2 = svc.fit(np.asarray(splitData[1]), np.asarray(splitTarget[1])).score(np.asarray(splitData[0]+splitData[2]), np.asarray(splitTarget[0]+splitTarget[2]))
    print(score2)
    score3 = svc.fit(np.asarray(splitData[2]), np.asarray(splitTarget[2])).score(np.asarray(splitData[1]+splitData[0]), np.asarray(splitTarget[1]+splitTarget[0]))
    print(score3)

main()
