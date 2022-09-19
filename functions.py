import numpy as np

def predict(X, Theta):
    return X @ Theta
def computeCost(X, y, Theta):
    predicted = predict(X, Theta)
    error = predicted - y
    m = np.size(y)
    J = (np.transpose(error) @ error) / (2*m)
    return J
def GradientDescent(X, y, alpha=0.02, iter=10000):
    Theta = np.zeros(np.size(X, 1))
    pre_cost = computeCost(X, y, Theta)
    m = np.size(y)
    X_T = np.transpose(X[:, 1:])
    for i in range(0, iter):
        predicted = predict(X, Theta)
        error = predicted - y
        tmp_b = Theta[0] - alpha * (np.sum(error)) / m
        tmp_w = Theta[1:] - alpha * (X_T @ error) / m
        Theta[0] = tmp_b
        Theta[1:] = tmp_w
        cost = computeCost(X, y, Theta)
        if (np.round(cost, 15) == np.round(pre_cost, 15)):
            print(Theta)
            break
        pre_cost = cost
    return Theta