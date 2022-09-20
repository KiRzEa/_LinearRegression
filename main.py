import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from functions import*
import matplotlib.pyplot as plt
#Simple Linear Regression
raw = np.loadtxt('data.txt', delimiter=',')
_X = np.copy(raw[:, :-1])
_X = normalize(_X)
_X = np.hstack((np.ones((_X.shape[0], 1)), _X))
_y = np.copy(raw[:, -1])
[_Theta, _J_hist] = GradientDescent(_X, _y)
print(_Theta)
plt.figure(1)
plt.plot(_J_hist[:, 0], _J_hist[:, 1], '-r')
#Multiple Linear Regression
boston = load_boston()
X = boston.data
y = boston.target
X = normalize(X)
X = np.hstack((np.ones((X.shape[0], 1)), X))
[Theta, J_hist] = GradientDescent(X, y, 0.1, 400)
print(Theta)
plt.figure(2)
plt.plot(J_hist[:, 0], J_hist[:, 1], '-b')
plt.show()
