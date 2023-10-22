import numpy as np
def mse(X, w, y_pred):
    y = X.dot(w)
    return (sum((y - y_pred)**2)) / len(y)

def calc_mse(y, y_pred):
    err = np.mean((y - y_pred)**2)
    return err
