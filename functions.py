import numpy as np
def predict(X, Theta):
    return X @ Theta
def normalize(X):
    return (X - np.mean(X, 0)) / np.std(X, 0)
def computeCost(X, y, Theta):
    error = predict(X, Theta) - y
    m = np.size(y)
    J = (np.transpose(error) @ error) / (2 * m)
    return J
def GradientDescent(X, y, alpha=0.02, iter=5000):
    Theta = np.zeros(np.size(X, 1), dtype=np.float64)
    pre_cost = computeCost(X, y, Theta)
    J_hist = np.zeros((iter, 2))
    X_T = np.transpose(X)
    m = np.size(y)
    for i in range(0, iter):
        error = predict(X, Theta) - y
        Theta = Theta - (alpha / m) * (X_T @ error)
        cost = computeCost(X, y, Theta)
        J_hist[i, 0] = i
        J_hist[i, 1] = cost
        if np.round(cost, 15) == np.round(pre_cost, 15):
            J_hist[i:, 0] = range(i, iter)
            J_hist[i:, 1] = cost
            break
        pre_cost = cost
    return Theta, J_hist
def NormalEquation(X, y):
    Theta = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    return Theta
