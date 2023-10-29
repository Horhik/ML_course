import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def mse(X, w, y_pred):
    y = X.dot(w)
    return (sum((y - y_pred)**2)) / len(y)

def calc_mse(y, y_pred):
    err = np.mean((y - y_pred)**2)
    return err

def bad_error_function(res, y):
    l = len(y)
    return (sum((res - y)**2)) / len(y)

def log_loss_sum(m, y, A):
    return (-1.0 / m * np.sum(
        y * np.log(A)
        + (1 - y) * np.log(1 - A)
    ))


def log_loss(w, X, y):
    m = X.shape[0]
    A = sigmoid(np.dot(X, w))
        
    # labels 0, 1
    loss = log_loss_sum(m, y, A)
    # labels -1, 1
#     temp_y = np.where(y == 1, 1, -1)
#     loss = 1.0 / m * np.sum(np.log(1 + np.exp(-temp_y * np.dot(X, w))))

    grad = 1.0 / m * X.T @ (A - y)

    return loss, grad

