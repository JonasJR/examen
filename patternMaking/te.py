from sklearn import datasets

digits = datasets.load_digits()
temp = str(len(digits.target))

print(temp)
