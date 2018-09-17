# predicting housing prices in boston
from sklearn.datasets import load_boston
import numpy as np
boston = load_boston()

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

# using 5th feature to predict
x = boston.data[:,5]
y= boston.target
x = np.transpose(np.atleast_2d(x)) # converts x from a one-dimensional to a two-dimensional array.
lr.fit(x,y)


from matplotlib import pyplot as plt
fig, ax = plt.subplots()
ax.scatter(x,y)

xmin= x.min()
xmax = x.max()

ax.plot([xmin, xmax], [lr.predict(xmin), lr.predict(xmax)], '-', color="#f9a602")
ax.set_xlabel("Average number of rooms (RM)")
ax.set_ylabel("House Price")
# plt.show()

# how close is our prediction
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, lr.predict(x))
print("Mean squared error (of training data): {:.3}".format(mse))

rmse = np.sqrt(mse)
print("RMSE (of training data): {:.3}".format(rmse))

# r2
from sklearn.metrics import r2_score
r2 = r2_score(y, lr.predict(x))
print("R2 (on training data): {:.2%}".format(r2))


