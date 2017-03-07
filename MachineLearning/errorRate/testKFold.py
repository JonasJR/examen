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

#we define a function for creating the randomized order of the dataset iris
#This function returns a new scrambled vector of the dataset
def randomize():
    new_dataset = []
    #We loop through the dataset and combine the data and target (category) into one element
    #and add it to the new_dataset vector
    for i in range(0,149):
        new_dataset.append([data[i],target[i]])
    #Now we shuffle the new_dataset vector so that the data is scrambled.
    new_dataset.shuffle()
    #And lastly we return the shuffled dataset
    return new_dataset

#We define a function to be used in the k-fold evaluation
#This function returns a score of the k-fold operation
def kFold():
    return 1

#We create a vector for storing the scores
scores = []

#We create a vector with the k-fold loops
loop = [2,3,5,6,10]
#We create the cross validator and store the scores
svc = SVC(kernel='linear', C=1)

#Counter for keeping track of the loops
counter = 0

#We create a loop that creates scores of the clf with incrementing test size 1% to 99%
for i in range(0,9):
    #We split the data into test and train with the given test_size
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=0)
    #We fit the svc and GaussianNB
    svc.fit(X_train, y_train)
    scores.append(svc.score(X_test, y_test))
    #scores.append(cross_val_score(clf, data,target))
    counter +=1

# #We create a plt to visualize the curves
# #Create the figure
# plt.figure()
# #Set title
# plt.title("")
# #set x and y lables
# plt.xlabel("Testing percentage")
# plt.ylabel("Score")
# #set it to grid style
# plt.grid()
# #set plot for svc and GaussianNB with coloring
# plt.plot(scores, 'o-', label="SVC", color="r", linestyle="--")
# #place the label in the top right
# plt.legend(loc="best")
# #show the figure
# plt.show()

temp = randomize()
print(str(temp))

#print(cv)
