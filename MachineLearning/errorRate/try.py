from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


iris = load_iris()
data, target = iris.data, iris.target

svc = SVC(kernel='linear', C=1)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.98, random_state=0)

svc.fit(X_train, y_train)

print(cross_val_score(svc,data, target, cv=50))
