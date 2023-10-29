import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from matplotlib.colors import ListedColormap



from errors import sigmoid
from gsd import make_gsd, eval_model
from plots import log_reg_plot


def standard_scale(x):
    res = (x - x.mean()) / x.std()
    return res

def categorize(value, border=0.5):
    categorized = value.copy()
    for i in range(value.shape[0]):
        if (value[i] > border): 
            categorized[i] = 1
        elif (value[i] <= border):
            categorized[i] = 0
    return categorized

def calc_pred_proba(X,W):
    predicted = X.dot(W)
    #print(" predicted is: ", predicted)
    return np.squeeze(sigmoid(predicted))


def calc_pred(X, W):
    return categorize(calc_pred_proba(X, W))

def eq (x,y):
    return x == y

def pairs(arr1, arr2):
    return np.array(list(map( tuple, np.array((arr1, arr2)).T)))

def compare(func, X, Y):
    return sum(map(func, pairs(X, Y)))
    


def Accuracy(initial, predicted):
    N = initial.shape[0]
    return sum(map(eq, pairs(initial, predicted)))/N

def error_matrix(Y, Y_pred):
   TP = compare(lambda v: v[0] == 1 and v[1] == 1, Y_pred, Y)
   FP = compare(lambda v: v[0] == 1 and v[1] == 0, Y_pred, Y)
   FN = compare(lambda v: v[0] == 0 and v[1] == 1, Y_pred, Y)
   TN = compare(lambda v: v[0] == 0 and v[1] == 0, Y_pred, Y)
   return (TP, FP, FN, TN)

def precision(e_matrix):
    (TP, FP, FN, TN) = e_matrix
    return TP/(TP + FP)

def recall(e_matrix):
    (TP, FP, FN, TN) = e_matrix
    return TP/(TP + FN)

def F_score(e_matrix, pay_respect_to_precision=1):

    prec = precision(e_matrix)
    rec = recall(e_matrix)
    multiplyer = (1 + pay_respect_to_precision**2)

    return multiplyer*prec*rec/(pay_respect_to_precision**2*prec + rec)



X_ = np.array([ [   1,    1,  500,    1],
               [   1,    1,  700,    1],
               [   1,    2,  750,    2],
               [   1,    5,  600,    1],
               [   1,    3, 1450,    2],
               [   1,    0,  800,    1],
               [   1,    5, 1500,    3],
               [   1,   10, 2000,    3],
               [   1,    1,  450,    1],
               [   1,    2, 1000,    2]], dtype=np.float64)

y = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.float64)


# Стандартизируем 3й столбец датасета
X = X_.copy()
X[:, 2] = standard_scale(X_[:, 2])


for i in np.linspace(1e-10, 1e-2, 100):
    (W, errors, wlist) = eval_model(X, y, N=3000, eta=i, logging_on=False)
    plt.plot(range(len(errors)), errors);
    neW = W.copy()
    y_pred = calc_pred(X, neW)

    #print(y_pred)
    e_matrix = error_matrix(y, y_pred)
    prec = precision(e_matrix)
    rec = recall(e_matrix)
    f = F_score(e_matrix)

print(" Y probabilitys is : ", calc_pred_proba(X, W))
print(" Error is : ", errors[-1])
print(" Precision is: ", prec)
print(" F-score is: ", f)
print(" recall is: ", rec)
print("Error Matrix is: ", e_matrix)


eta = 0.05
#(y_tested_pred, w_test, losses_tested, test_accuracy) =  make_gsd(X_test, y_test, n_iterations, eta)

#rint(f"Точность на тестовой выборке: {test_accuracy:.3f}")

plt.title('Log loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.plot(range(len(errors)), errors);
plt.show()
plt.plot(range(len(wlist)), wlist);
plt.show()


