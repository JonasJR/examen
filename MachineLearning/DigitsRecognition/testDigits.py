from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

#Loads the digit dataset
digits = load_digits()

#stores all the data in X and all target in y
X = digits.data
y = digits.target

#Uses K-Nearest Neighbor to classify. We use n_neighbors=5 to broaden the spectrum
#It calculates the number of closest neighbors, and in this case it's 5
#If we want we can change it to 1 to get a closer match
knn = KNeighborsClassifier(n_neighbors=5)

#We fit with data
knn.fit(X, y)

#We pick a randon value for testing (even tough it alreddy learned it)
X_test = X[456]

#Prints the prediption to the screen
print knn.predict(X_test.reshape(1, -1))
