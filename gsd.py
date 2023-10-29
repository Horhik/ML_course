import numpy as np
from errors import mse, calc_mse, log_loss, sigmoid
import matplotlib.pyplot as plt




def dl1_regulation(func):
  def wrapper(*args, **kwargs):
    return func(*args, **kwargs, regulator=lambda l,w: l*sum(w/np.abs(w)))
  return wrapper


def dl2_regulation(func):
  def wrapper(*args, **kwargs):
    return func(*args,**kwargs, regulator=lambda l,w: 2*l*np.sum(w))
  return wrapper


def no_regulation(func):
  def wrapper(*args):
    return func(*args, regulator=lambda l,w: 0)
  return wrapper

    



def stohastic_gsd(X, Y, eta=0.01, max_iter = 1e3, reg=0,  regulator=lambda l,w: 0):
    # инициализируем начальный вектор весов
    w = np.zeros(X.shape[1])

    # список векторов весов после каждой итерации
    w_list = [w.copy()]
    weights_change = []

    # список значений ошибок после каждой итерации
    errors = []


    # максимальное число итераций

    # критерий сходимости (разница весов, при которой алгоритм останавливается)
    min_weight_dist = 1e-8

    # зададим начальную разницу весов большим числом
    weight_dist = np.inf
    
    # счетчик итераций
    iter_num = 0

    np.random.seed(1234)
    # ход градиентного спуска
    while weight_dist > min_weight_dist and iter_num < max_iter:
    
        # генерируем случайный индекс объекта выборки
        i = np.random.randint(X.shape[0], size=1)
        l = Y[i].shape[0] #Y[i].shape[0]
    
        y_pred = np.dot(X[i], w)

        y_change = y_pred - Y[i]
        dQi = 2/l * np.dot(X[i].T, y_change)
        dRegi = regulator(reg, [w])

        new_w = w - eta * (dQi + dRegi) # making a step with regulation
        
        weight_dist = np.linalg.norm(new_w - w, ord=2)
        weights_change.append(weight_dist)
        
        error = mse(X, new_w, Y)

        
        w_list.append(new_w.copy())
        errors.append(error)
        
        if iter_num % 100 == 0:
            print(f'Iteration #{iter_num}: W_new = {new_w}, MSE = {round(error, 2)}')
            
        iter_num += 1
        w = new_w


    print(f'Iter {iter_num}: error - {error}, weights: {new_w}')
    print(f'В случае использования стохастического градиентного спуска ошибка составляет {round(errors[-1], 4)}')

    return(w_list, weights_change,  errors)


def gsd(X, y, eta=1e-3, max_iter=1e3, reg=0, regulator=lambda reg, w: 0):
    
    max_iter = 1e3
    
    epsilon = 1e-8
    
    w_change = 100000
    

    W = np.ones(X.shape[1])
    i = 0
    #buffers for return info

    errors = []
    weights_change = []
    weights = []

    while w_change > epsilon and i < max_iter: # Движение градиентного спуска до тех пор пока не наступит сходимость или не превысется количество итераций
        y_pred = X @ W
        
        err = calc_mse(y, y_pred)

        errors.append(err)
        
        w_old = W.copy()
        #for k in range(W.shape[0]):
        #    W[k] -= eta * (1/n * 2 * X[:, k] @ (y_pred - y))
        dQ = (1/len(y) * 2 *np.dot(X.T , (y_pred - y)))
        #print("dQ change is: ", dQ)
        dReg = regulator(reg, W)
        #print("dReg change is: ", dReg)
        W -= eta *(dQ + dReg) 
        #print("W change is: ", W)
        
        # counting weights between distance
        w_change = np.linalg.norm(W - w_old)
        weights.append(w_old)
        
        weights_change.append(w_change)
        if i % 1 == 0:
            eta /= 1.1
            print(f'Iteration #{i}: W_new = {W}, MSE = {round(err, 2)}')
            print(f'Change is {w_change}')
        i+=1

        #print(w_change > epsilon)
        #print(i < max_iter)
        #print(w_change , epsilon)
    return(weights, weights_change, errors)
    
            


stohastic_gsd_l1 = dl1_regulation(stohastic_gsd)
stohastic_gsd_l2 = dl2_regulation(stohastic_gsd)

gsd_l1 = dl1_regulation(gsd)
gsd_l2 = dl2_regulation(gsd)


def optimize(w, X, y, n_iterations, eta):
    # потери будем записывать в список для отображения в виде графика
    losses = []
    
    for i in range(n_iterations):        
        loss, grad = log_loss(w, X, y)
        w = w - eta * grad

        losses.append(loss)
        
    return w, losses

def predict(w, X):
    
    m = X.shape[0]
    
    y_predicted = np.zeros(m)

    A = np.squeeze(sigmoid(np.dot(X, w)))

    # За порог отнесения к тому или иному классу примем вероятность 0.5
    for i in range(A.shape[0]):
        if (A[i] > 0.5): 
            y_predicted[i] = 1
        elif (A[i] <= 0.5):
            y_predicted[i] = 0

    return y_predicted


def make_gsd(X, Y, N, eta):
  w0 = np.zeros(X.shape[1])
  w, losses = optimize(w0, X, Y, N, eta)
  y_predicted = predict(w, X)
  train_accuracy = np.mean(y_predicted == Y) * 100.0


  return (y_predicted, w, losses, train_accuracy)
  
