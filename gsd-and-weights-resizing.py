import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Local imports
from plots import weights_change, error_change
from gsd import gsd, stohastic_gsd, stohastic_gsd_l1, stohastic_gsd_l2, gsd_l1, gsd_l2


#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# create dataset
X, Y, coef = datasets.make_regression(n_samples=1000, n_features=2, n_informative=2, n_targets=1, 
                                      noise=5, coef=True, random_state=2)
X[:, 0] *= 10

'''
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], Y)

ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_zlabel('Y')
plt.show()
'''

# Получим средние значения и стандартное отклонение по столбцам
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)

resizedX = np.array([[(X[i][j] - means[j]) / stds[j] for j in range(X.shape[1])] for i in range(X.shape[0])])

print(np.mean(resizedX, axis=0))
print(np.std(resizedX, axis=0))


# evaluating GSD
(w_list,w_change, errors) = gsd(resizedX, Y, reg = 1)
w_list = np.array(list((map(list, w_list))))

gsd_W = w_list[-1]

#weights_change(w_list, w_list[-1])
#error_change(errors)
#error_change(w_change)


(w_list,w_change, errors) = stohastic_gsd(resizedX, Y, reg = 1)
w_list = np.array(list((map(list, w_list))))

gsdsW = w_list[-1]


x1 = np.linspace(-10, 10)
x2 = np.linspace(-10, 10)
xx, yy = np.meshgrid(x1, x2, indexing='ij')

y_gsd = []
y_gsds = []
z1 = gsdsW[0]*xx + gsdsW[1]*yy
z2 = gsd_W[0]*xx + gsd_W[1]*yy


print(len(x1))
print(len(x2))
print(len(y_gsd))


        
#y_pred_grad_stoh = X@stohastic_W
#y_pred_grad = X@gsd_W

#plt.scatter(X[:, 1], Y)
#plt.scatter(X[:, 1], y_pred_grad)
#plt.scatter(X[:, 1], y_pred_grad_stoh)

#fig = plt.figure(figsize=(15,10))
#ax = fig.add_subplot(111, projection='3d')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(X[:, 0], X[:, 1], Y)
ax.plot_surface(xx, yy, z1)
ax.plot_surface(xx, yy, z2)

#ax.plot(xx, yy, z1)
#ax.plot(xv, yv, y_gsd)





#ax.scatter(X[:, 0], X[:, 1], y_pred_grad)
#ax.scatter(X[:, 0], X[:, 1], y_pred_grad_stoh)
plt.show()
'''
#plt.subplot(211)
#plt.plot(X[:, 1], Y, label='3 - analytical solution')
plt.plot(X[:, 0], y_pred_grad, label='gradient descent', c='r')
plt.plot(X[:, 0], y_pred_grad_stoh, label='gradient descent stohastic', c='r')




plt.legend()
plt.show()

'''
