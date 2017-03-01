from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Loads the digit dataset
digits = load_digits()

#stores all the data in X and all target in y
X = digits.data
y = digits.target

train_data, test_data, train_target, test_target = train_test_split(X, y, random_state=42, test_size=0.95)

#Uses K-Nearest Neighbor to classify. We use n_neighbors=5 to broaden the spectrum
#It calculates the number of closest neighbors, and in this case it's 5
#If we want we can change it to 1 to get a closer match
knn = KNeighborsClassifier(n_neighbors=1)

#We fit with data
knn.fit(train_data, train_target)

#We pick a randon value for testing (even tough it alreddy learned it)
#X_test = len(X)

print "True data: " + str(test_target)

#Prints the prediption to the screen
print "Prediction: " + str(knn.predict(test_data))
