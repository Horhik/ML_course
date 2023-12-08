import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

X, y = load_iris(return_X_y=True)

# Для наглядности возьмем только первые два признака (всего в датасете их 4)
X = X[:, :2]

# Создадим тестовую и обучающую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y)
print(" Y TRAIN IS: ", y_train)
cmap = ListedColormap(['red', 'green', 'blue'])
plt.figure(figsize=(7, 7))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap);

def e_metrics(x1, x2):
    
    distance = np.sum(np.square(x1 - x2))

    return np.sqrt(distance)

def count_weight(iteration, metric, q=0.99, a = 1):
    return 1- 1/(metric + a)**2

    #1 - 1/(iteration+1) + 1- 1/(metric + a)**2
    #return q**(iteration+1) + q**metric

def knn_2(x_train, y_train, x_test, k):
    start_time = time.time()
    answers = []
    for x in x_test:
        test_distances = []
        test_weights = []
            
        global_counter = 0
            
            # расчет расстояния от классифицируемого объекта до
            # объекта обучающей выборки
            #distance = e_metrics(x, x_train[i])
            # и высчитывание весов на осонвании числа итераций 
        distances = np.array([count_weight(global_counter + i, e_metrics(x, x_train[i])) for i in range(len(x_train))])
        global_counter += len(x_train)
        weights = count_weight(global_counter, distances)
        sorted_indices = np.argsort(weights)[:k]
        counts = np.bincount(y_train[sorted_indices])
        answers.append(np.argmax(counts))

            
        
        # создаем словарь со всеми возможными классами
        #classes = {class_item: 0 for class_item in set(y_train)}
        
        # Сортируем список и среди первых k элементов подсчитаем частоту появления разных классов
        #print(weights)
        #for d in sorted(weights)[0:k]:
        #    classes[d[1]] += 1

        # Записываем в список ответов наиболее часто встречающийся класс
        #answers.append(sorted(classes, key=classes.get)[-1])
    end_time = time.time()
    print(" KNN2 took ", end_time - start_time)
        
    return answers

def knn(x_train, y_train, x_test, k):
    
    start_time = time.time()
    answers = []
    for x in x_test:
        test_distances = []
        test_weights = []
            
        global_counter = 0
        for i in range(len(x_train)):
            
            # расчет расстояния от классифицируемого объекта до
            # объекта обучающей выборки
            distance = e_metrics(x, x_train[i])

            weight = count_weight(global_counter, distance)
            
            # Записываем в список значение расстояния и ответа на объекте обучающей выборки
            test_distances.append((distance, y_train[i]))
            test_weights.append((weight, y_train[i]))
            global_counter+=1
        
        # создаем словарь со всеми возможными классами
        classes = {class_item: 0 for class_item in set(y_train)}
        
        # Сортируем список и среди первых k элементов подсчитаем частоту появления разных классов
        for d in sorted(test_weights)[0:k]:
            classes[d[1]] += 1

        # Записываем в список ответов наиболее часто встречающийся класс
        answers.append(sorted(classes, key=classes.get)[-1])
        
    end_time = time.time()
    print(" KNN took ", end_time - start_time)
    return answers



def knn_3(x_train, y_train, x_test, k):
    
    start_time = time.time()
    answers = []
    global_counter = 0
    for x in x_test:
        test_distances = []
        test_weights = []
            
        first_k = []
        minimum = 100000000
        for i in range(len(x_train)):
            global_counter +=1
            
            # расчет расстояния от классифицируемого объекта до
            # объекта обучающей выборки
            distance = e_metrics(x, x_train[i])

            weight = count_weight(global_counter, distance)
            if(weight < minimum):
                minimum = weight
                first_k.append((minimum, y_train[i]))
            
            # Записываем в список значение расстояния и ответа на объекте обучающей выборки
            test_distances.append((distance, y_train[i]))
            test_weights.append((weight, y_train[i]))
        
        # создаем словарь со всеми возможными классами
        classes = {class_item: 0 for class_item in set(y_train)}
        
        # Сортируем список и среди первых k элементов подсчитаем частоту появления разных классов
        for d in first_k[-k:]:
            classes[d[1]] += 1

        # Записываем в список ответов наиболее часто встречающийся класс
        answers.append(sorted(classes, key=classes.get)[-1])
        
    end_time = time.time()
    print(" KNN3 took ", end_time - start_time)
    return answers

def accuracy(pred, y):
    return (sum(pred == y) / len(y))

k = 5

y_pred = knn(X_train, y_train, X_test, k)
y_pred = knn_2(X_train, y_train, X_test, k)
y_pred = knn_3(X_train, y_train, X_test, k)

print(f'Точность алгоритма при k = {k}: {accuracy(y_pred, y_test):.3f}')

def get_graph(X_train, y_train, k):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])

    h = .1

    # Расчет пределов графика
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    print(np.c_[xx.ravel(), yy.ravel()].shape)

    # Получим предсказания для всех точек
    Z = knn(X_train, y_train, np.c_[xx.ravel(), yy.ravel()], k)
    # Построим график
    Z = np.array(Z).reshape(xx.shape)
    plt.figure(figsize=(7,7))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Добавим на график обучающую выборку
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"Трехклассовая kNN классификация при k = {k}")
    plt.show()

get_graph(X_train, y_train, k)
