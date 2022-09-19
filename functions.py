import numpy as np
def predict(X, Theta):
    return X @ Theta
def computeCost(X, y, Theta):
    error = predict(X, Theta) - y
    m = np.size(y)
    J = (np.transpose(error) @ error) / (2 * m)
    return J
def normalize(X):
    return (X - np.mean(X)) / (np.std(X))
def GradientDescent(X, y, alpha=0.01, iter=10000):
    Theta = np.zeros(np.size(X, 1))
    pre_cost = computeCost(X, y, Theta)
    X_T = np.transpose(X)
    m = np.size(y)
    for i in range(0, iter):
        error = predict(X, Theta) - y
        Theta = Theta - (alpha / m) * (X_T @ error)
        cost = computeCost(X, y, Theta)
        if (np.round(cost, 15) == np.round(pre_cost, 15)):
            break
        pre_cost = cost
    return Theta
def NormalEquation(X, y):
    Theta = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    return Theta
