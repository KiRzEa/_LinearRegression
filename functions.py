import numpy as np
def predict(X, Theta):
    return X @ Theta
def computeCost(X, y, Theta):
    sqr_error = (predict(X, Theta) - y) ** 2
    m = np.size(y)
    J = 1.0/(2*m) * np.sum(sqr_error)
    return J
def GradientDescent(X, y, alpha=0.00001, iter=10000):
    Theta = np.zeros(np.size(X, 1))
    pre_cost = computeCost(X, y, Theta)
    X_T = np.transpose(X[:, 1:])
    m = np.size(y)
    for i in range(0, iter):
        error = predict(X, Theta) - y
        tmp_b = Theta[0] - (alpha / m) * (np.sum(X[:, 0] @ error))
        tmp_w = Theta[1:] - (alpha / m) * (X_T @ error)
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
