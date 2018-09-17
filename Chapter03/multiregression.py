# predicting housing prices in boston
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
boston = load_boston()

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

x = boston.data
y = boston.target
lr.fit(x,y)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, lr.predict(x))
print("Mean squared error (on training data): {:.3}".format(mse))
rmse = np.sqrt(mse)
print("RMSE (on training data): {}".format(rmse))
r2 = r2_score(y, lr.predict(x))
print("R2 (on training data): {:.2}".format(r2))


p = lr.predict(x)
fig, ax = plt.subplots()

ax.set_xlabel('Predicted price')
ax.set_ylabel('Actual Price')
ax.plot([y.min(), y.max()], [[y.min()], [y.max()]], ":", lw=2, color="#f9a602")
ax.scatter(lr.predict(x), y, s= 2)
fig.savefig("Regression_Fig_02.png")

# cross validation
from sklearn.model_selection import KFold, cross_val_predict
kf = KFold(n_splits=5)
p = cross_val_predict(lr, x,y,cv=kf)
rmse_cv = np.sqrt(mean_squared_error(p ,y))
print("RMSE on 5-fold CV: {:.2}".format(rmse_cv))

# penalized regression
# adds a penalty for over confidence in the parameter values
# accept a slightly worse fit in order to have a simpler model.
# helps with over fitting, might generalize better to unseen (test) data.
