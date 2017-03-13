from sklearn.datasets import load_digits, load_iris
from sklearn.svm import SVC
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import numpy as np
import math
import random

#########   Note for IVAN!!!!   ############
# Just an intro to how python works. A python program works kind of like a C-program.
# It starts by reading from the top and executing everything it encounters on the way.
# If it encounters something that has not yet been declared it will generate an error.
# We can execute commands directly in the code and create functions that we later call.
# The easiest is therefore to define all the functions in the beginning and then execute commands.

#We start of by loading the iris datasets and store them in data and target
iris = load_iris()
data, target = iris.data, iris.target
#We create the cross validator
#svc = linear_model.LinearRegression()
svc = SVC(kernel='linear', C=1.0)
#svc = tree.DecisionTreeClassifier()

# To simplify the looping later on I stole some code online that generates an array
# of "possible divisions" of a given number. So passing in 150 would yeild
# following result: [1,2,3,5,6,10,15,25,30,50,75,150]
# We will remove the first and last item in the list since we are not using them
def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n/i)
    for divisor in reversed(large_divisors):
        yield divisor

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

#the data array becomes corrupted when we are mixtering with it (this is where I got stuck)
#so we have to resolve it so it can be used by the svc.fit()
#We define a function that resolves the array and returns a usable array
def resolve(a):
    temp = []
    for i in a:
        for e in i:
            temp.append(e)
    return np.asarray(temp)

#We define a function to be used in the k-fold evaluation
#This function returns a score of the k-fold operation
#n is the amount of splits and d is the vector that should be split
def kFold(n, d):
    #Here we pass an iterator (the dataset) to the zip function n times. Each iteration it pulls an item from the dataset
    #If n=3 the result is a vector with 3 elements.
    return zip(*[iter(d)]*n)

#We define a function that gets the score of the kfold
#we loop through the splits and for each split we
#calculate the score and create the avarage score
def getScore(data,target,i):
    #First we create a temporary vector to store all scores for later avarage calculation
    tempScores = cross_val_score(svc,data,target,cv=i)
    print(tempScores)
    #We loop through the splits
    #Now we just have to get the avarage of the score and then return it
    return reduce(lambda x, y: x + y, tempScores) / len(tempScores)
    #Test to return lowest score:
    #return min(float(s) for s in tempScores)

#We define a function that loops through the kfold split array and gets the scores and create a median of it.
#v is the vector that was split by the kfold function
def main():
    #here we just get the list of divisors for the looping and remove first and last items
    loop = list(divisorGenerator(len(data)))
    loop.pop(0)
    loop.pop(len(loop)-1)
    loop.pop(len(loop)-1)
    print(loop)
    #We start with calling the randomize function to make sure that the data is shuffled
    randomize()
    #We create a vector for storing the scores
    scores = []
    #We create a loop that loops through the lenght of the dataset devided by two.
    #This makes the last iteration create kfold with 2 elements in each group. (2 training data and 148 test data)
    #We start at 2 because 0 and 1 is not possible when doing a kfold
    for i in loop:
        #We the get cross_val_score of the datase
        score = getScore(data,target, i)
        #And add the calculated score to the scores vector
        scores.append(score)
        break
    #print(scores)

    #We create a plt to visualize the curves
    #Create the figure
    plt.figure()
    #Set title
    plt.title("")
    #set x and y lables
    plt.xlabel("Size of K-fold")
    plt.ylabel("Score")
    #set it to grid style
    plt.grid()
    #set plot for svc and GaussianNB with coloring
    plt.plot(scores, 'o-', label="SVC cross_val_score iris", color="r", linestyle="--")
    #place the label in the top right
    plt.legend(loc="best")
    #show the figure
    plt.show()

main()
print("done")
