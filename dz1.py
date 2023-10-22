import matplotlib.pyplot as plt
import numpy as np
X = np.array([[ 1,  1],
              [ 1,  1],
              [ 1,  2],
              [ 1,  5],
              [ 1,  3],
              [ 1,  0],
              [ 1,  5],
              [ 1, 10],
              [ 1,  1],
              [ 1,  2],
              [ 1,  1],
              [ 1,  20],
              ]
             )

X.shape
y = [45, 55, 50, 55, 60, 35, 75, 80, 50, 60, 30, 99]
y_pred1 = 5 * X[:, 1] + 35 * X[:, 0] 
y_pred2 = 7.5 * X[:, 1] + 40 * X[:, 0]
#plt.plot(X[:, 1], y_pred1, label='1')
#plt.plot(X[:, 1], y_pred2, label='2')
plt.legend()

err1 = np.sum(y - y_pred1)
err2 = np.sum(y - y_pred2)

mae_1 = np.mean(np.abs(y - y_pred1))
mae_2 = np.mean(np.abs(y - y_pred2))

mse_1 = np.mean((y - y_pred1)**2)
mse_2 = np.mean((y - y_pred2)**2)

W_analytical = np.linalg.inv(np.dot(X.T, X)) @ X.T @ y

y_pred_analytical = W_analytical[0] * X[:, 0] + W_analytical[1] * X[:, 1]
y_pred_analytical = X @ W_analytical

#plt.plot(X[:, 1], y_pred1, label='1 - manual')
#plt.plot(X[:, 1], y_pred2, label='2 - manual')
#plt.show()

def calc_mae(y, y_pred):
    err = np.mean(np.abs(y - y_pred))
    return err

def calc_mse(y, y_pred):
    err = np.mean((y - y_pred)**2)
    return err

calc_mae(y, y_pred1), calc_mse(y, y_pred1)

calc_mae(y, y_pred_analytical), calc_mse(y, y_pred_analytical)

W = np.random.normal(size=(X.shape[1]))
W

eta = 0.02 # величина шага

X.shape,  W.shape

n = len(y)
dQ = 2/n * X.T @ (X @ W - y) # градиент функции ошибки
dQ

grad = eta * dQ
grad

print(f'previous weights', W)
W = W - grad
print(f'new weights', W)

y_pred_grad = X @ W
#plt.plot(X[:, 1], y_pred_grad, label='gradient descent', c='g')


n = X.shape[0]

eta = 35.09e-3
n_iter = 500

epsilon = 1e-1

w_change = epsilon*100000

W = np.array([1, 0.5])
print(f'Number of objects = {n} \
       \nLearning rate = {eta} \
       \nInitial weights = {W} \n')
i = 0
errors = []
errors_mae = []
weight_change = []
while w_change > epsilon and i < n_iter: # Движение градиентного спуска до тех пор пока не наступит сходимость или не превысется количество итераций
    print(i)
    y_pred = X @ W
    err = calc_mse(y, y_pred)
    err_mae = calc_mae(y, y_pred)
    errors.append(err)
    errors_mae.append(err_mae)
    w_old = W.copy()
    #for k in range(W.shape[0]):
    #    W[k] -= eta * (1/n * 2 * X[:, k] @ (y_pred - y))
    W -= eta * (1/n * 2 *(np.matrix.transpose(X) @ (y_pred - y))) #Матрицу нужно транспонировать 
    #ch_vec = W - w_old
    #w_change = np.sqrt(ch_vec[1]**2 - ch_vec[0]**2)

    # counting weights between distance
    w_change = np.linalg.norm(W - w_old)

    weight_change.append(w_change)
    if i % 10 == 0:
        eta /= 1.1
        print(f'Iteration #{i}: W_new = {W}, MSE = {round(err, 2)}')
    print(f'Change is {w_change}')
    i+=1
        
y_pred_grad = X@W

plt.subplot(211)
plt.scatter(X[:, 1], y)
plt.plot(X[:, 1], y_pred_analytical, label='3 - analytical solution')
plt.plot(X[:, 1], y_pred_grad, label='gradient descent', c='r')
plt.legend()



plt.subplot(212)
plt.plot(np.linspace(1, len(errors), len(errors)), errors/np.max(errors), label='error change', c='r')

plt.plot(np.linspace(1, len(errors_mae), len(errors_mae)), errors_mae/np.max(errors_mae), label='Mae change', c='g')

plt.plot(np.linspace(1, len(weight_change), len(weight_change)), weight_change/np.max(weight_change), label='Weight change', c='b')

plt.legend()


plt.show()
