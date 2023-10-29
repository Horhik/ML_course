import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from matplotlib.colors import ListedColormap


import errors
from gsd import make_gsd
from plots import log_reg_plot

# сгеренируем данные с помощью sklearn.datasets
X, y = datasets.make_classification(n_samples=100, n_features=2, n_informative=2,
                                    n_redundant=0, n_classes=2, random_state=1)
X, y = datasets.make_blobs(centers=2, cluster_std=2.5, random_state=12)

# и изобразим их на графике
colors = ListedColormap(['blue', 'red'])

#plt.figure(figsize=(8, 8))
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colors);

np.random.seed(12)
shuffle_index = np.random.permutation(X.shape[0])
X_shuffled, y_shuffled = X[shuffle_index], y[shuffle_index]

# разбивка на обучающую и тестовую выборки
train_proportion = 0.7
train_test_cut = int(len(X) * train_proportion)

X_train, X_test, y_train, y_test = \
    X_shuffled[:train_test_cut], \
    X_shuffled[train_test_cut:], \
    y_shuffled[:train_test_cut], \
    y_shuffled[train_test_cut:]
    
print("Размер массива признаков обучающей выборки", X_train.shape)
print("Размер массива признаков тестовой выборки", X_test.shape)
print("Размер массива ответов для обучающей выборки", y_train.shape)
print("Размер массива ответов для тестовой выборки", y_test.shape)


n_iterations = 10000
eta = 0.05
(y_trained_pred, w_train, losses_trained, train_accuracy) =  make_gsd(X_train, y_train, n_iterations, eta)
(y_tested_pred, w_test, losses_tested, test_accuracy) =  make_gsd(X_test, y_test, n_iterations, eta)


print(f"Итоговый вектор весов w: {w_train}")
print(f"Точность на обучающей выборке: {train_accuracy:.3f}")
print(f"Точность на тестовой выборке: {test_accuracy:.3f}")

plt.title('Log loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.plot(range(len(losses_trained)), losses_trained);
plt.show()


log_reg_plot(X, w_train, y, colors)
plt.show()
