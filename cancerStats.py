import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = pd.read_csv("cancer.csv")

# build correlation matrix
corr = data.corr()
print("Correlation Matrix:")
print(corr)

# drop highly independent correlated attributes
data = data.drop(['perimeter', 'area'], axis=1)

# show stats
print("Data Stats:")
print(data.describe())

# scatter & box plot
# pd.plotting.scatter_matrix(data)
plt.scatter(data["radius"], data["diagnosis"])
plt.xlabel("Radius")
plt.ylabel("Diagnosis")
plt.show()

plt.boxplot(data["radius"], vert=False, meanline=True, showmeans=True)
plt.show()

plt.scatter(data["smoothness"], data["diagnosis"])
plt.xlabel("Smoothness")
plt.ylabel("Diagnosis")
plt.show()

plt.boxplot(data["smoothness"], vert=False, meanline=True, showmeans=True)
plt.show()

plt.scatter(data["texture"], data["diagnosis"])
plt.xlabel("Texture")
plt.ylabel("Diagnosis")
plt.show()

plt.boxplot(data["texture"], vert=False, meanline=True, showmeans=True)
plt.show()

# spilt data to 80% training 20% test
X = data.drop(['diagnosis'], axis=1)
y = data['diagnosis']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1, test_size=0.4)

# build regression model
linearRegression = LinearRegression()
linearRegression.fit(Xtrain, ytrain)

# print model coefficients
print("Model Coefficients:")
print(linearRegression.intercept_)
print(linearRegression.coef_)

# test model using the test data set
ypredicted = linearRegression.predict(Xtest)

xr = np.arange(1, 21)
plt.scatter(xr, ypredicted[:20], color='b')
plt.scatter(xr, ytest[:20], color='g')
plt.plot([1, 21], [0.4, 0.4], color="r")
plt.plot([1, 21], [0.6, 0.6], color="r")
plt.ylabel("Diagnosis")
plt.show()

r2 = r2_score(ytest, ypredicted)
print("R Squared:")
print("R2 = {:.2}".format(r2))
