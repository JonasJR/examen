from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()
type(digits)

X = digits.data
y = digits.target

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X, y)

n_samples = len(X)

X_test = X[456]

print knn.predict(X)
