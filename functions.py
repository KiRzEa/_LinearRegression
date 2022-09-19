import numpy as np


def predict(X, Theta):
    return X @ Theta


def computeCost(X, y, Theta):
    predicted = predict(X, Theta)
    error = predicted - y
    m = np.size(y)
    J = (np.transpose(error) @ error) / (2 * m)
    return J


def GradientDescent(X, y, alpha=0.01, iter=10000):
    Theta = np.zeros(np.size(X, 1))
    pre_cost = computeCost(X, y, Theta)
    X_T = np.transpose(X[:, 1:])
    for i in range(0, iter):
        predicted = predict(X, Theta)
        error = predicted - y
        tmp_b = Theta[0] - alpha * (np.mean(error))
        tmp_w = Theta[1:] - alpha * (np.mean(X_T @ error))
        Theta[0] = tmp_b
        Theta[1:] = tmp_w
        cost = computeCost(X, y, Theta)
        if (np.round(cost, 15) == np.round(pre_cost, 15)):
            break
        pre_cost = cost
    return Theta
def GradientDescent_1(X, y):
    Theta = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    return Theta
