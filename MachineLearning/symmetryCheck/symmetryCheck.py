import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("trainingdata.csv")

X = data[['Row1','Row2','Row3','Row4']]

y = data['Sym']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)
