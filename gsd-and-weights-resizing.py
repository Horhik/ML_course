import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Local imports
from plots import weights_change, error_change
from gsd import gsd, stohastic_gsd, stohastic_gsd_l1, stohastic_gsd_l2, gsd_l1, gsd_l2


import warnings
warnings.filterwarnings('ignore')

# create dataset
X, Y, coef = datasets.make_regression(n_samples=1000, n_features=2, n_informative=2, n_targets=1, 
                                      noise=5, coef=True, random_state=2)
X[:, 0] *= 10

# Получим средние значения и стандартное отклонение по столбцам
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)


def normalize(X):
    maX = np.max(X)
    miN = np.min(X)
    resizedX = np.array([[(X[i][j] - miN) / (maX - miN) for j in range(X.shape[1])] for i in range(X.shape[0])])
    return resizedX

def standartize(X):
    mu = 1/len(X)*np.sum(X) 
    sig = np.sqrt(1/len(X) * np.sum(X - mu)**2)
    resizedX = np.array([[(X[i][j] -mu)/sig for j in range(X.shape[1])] for i in range(X.shape[0])])
    return resizedX



resizedX = standartize(normalize(X))

print(np.mean(resizedX, axis=0))
print(np.std(resizedX, axis=0))


# evaluating GSD
(w_list,w_change, errors) = gsd(resizedX, Y, reg = 0, max_iter = 1e3)
w_list = np.array(list((map(list, w_list))))

gsd_W = w_list[-1]


(w_list,w_change_s, errors_s) = stohastic_gsd(resizedX, Y, reg = 0, max_iter = 1e3, eta = 1e-2)
w_list = np.array(list((map(list, w_list))))

gsdsW = w_list[-1]


x1 = np.linspace(-10, 10)
x2 = np.linspace(-10, 10)
xx, yy = np.meshgrid(x1, x2, indexing='ij')

y_gsd = []
y_gsds = []
z1 = gsdsW[0]*xx + gsdsW[1]*yy
z2 = gsd_W[0]*xx + gsd_W[1]*yy

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(X[:, 0], X[:, 1], Y)

ax.plot_surface(xx, yy, z1)
ax.plot_surface(xx, yy, z2)

plt.show()

error_change(errors)
error_change(errors_s)
plt.show()
