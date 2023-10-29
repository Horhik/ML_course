import matplotlib.pyplot as plt
import numpy as np
from gsd import predict

def weights_change(w_list, w_final):
# Визуализируем изменение весов (красной точкой обозначены истинные веса, сгенерированные вначале)
    plt.figure(figsize=(13, 6))
    plt.title('Stochastic gradient descent')
    plt.xlabel(r'$w_1$')
    plt.ylabel(r'$w_2$')
    
    plt.scatter(w_list[:, 0], w_list[:, 1])
    plt.scatter(w_final[0], w_final[1], c='r')
    plt.plot(w_list[:, 0], w_list[:, 1])
    
    plt.show()
def error_change(errors):
    # Визуализируем изменение функционала ошибки
    plt.plot(range(len(errors)), errors)
    plt.title('MSE')
    plt.xlabel('Iteration number')
    plt.ylabel('MSE')


def log_reg_plot(X, w, y, colors):
   plt.figure(figsize=(8, 8))

   x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
   y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
   h = .02  # step size in the mesh
   xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
   Z = predict(w, np.c_[xx.ravel(), yy.ravel()])

   # Put the result into a color plot
   Z = Z.reshape(xx.shape)
   plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

   plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colors);
