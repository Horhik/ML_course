from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import numpy as np

X, y = load_diabetes(return_X_y=True)
X.shape, y.shape

# Разделим выборку на обучающую и тестовую в соотношении 75/25.
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

def mean_squared_error(y_real, prediction):
    return (sum((y_real - prediction)**2)) / len(y_real)

def residual(y, z):
    #derivative of L(y,z) = (y-z)^2 without  multiplying to 2
    return - (z - y)

def gb_predict(X, trees_list, eta):
    predictions = np.zeros(X.shape[0])
    
    for tree in trees_list:
        predictions += eta * tree.predict(X)

    return predictions

def _gb_predict(X, trees_list, eta):
    # Реализуемый алгоритм градиентного бустинга будет инициализироваться нулевыми значениями,
    # поэтому все деревья из списка trees_list уже являются дополнительными и при предсказании
    # прибавляются с шагом eta
    
    predictions = np.zeros(X.shape[0])
    for i, x in enumerate(X):
        prediction = 0
        for alg in trees_list:
            prediction += eta * alg.predict([x])[0] # <- можно скормить весь массив!!
        predictions[i] = prediction
        
    predictions = np.array(
        [sum([eta * alg.predict([x])[0] for alg in trees_list]) for x in X]
    )

    return predictions



def gb_fit(n_trees, max_depth, X_train, X_test, y_train, y_test, eta):
    
    # Деревья будем записывать в список
    trees = []
    
    # Будем записывать ошибки на обучающей и тестовой выборке на каждой итерации в список
    train_errors = []
    test_errors = []
    
    for i in range(n_trees):
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)

        # первый алгоритм просто обучаем на выборке и добавляем в список
        if len(trees) == 0:
            # обучаем первое дерево на обучающей выборке
            tree.fit(X_train, y_train)
            
            train_errors.append(mean_squared_error(y_train, gb_predict(X_train, trees, eta)))
            test_errors.append(mean_squared_error(y_test, gb_predict(X_test, trees, eta)))
        else:
            # Получим ответы на текущей композиции
            target = gb_predict(X_train, trees, eta)
            
            # алгоритмы начиная со второго обучаем на сдвиг
            tree.fit(X_train, residual(y_train, target))
            
            train_errors.append(mean_squared_error(y_train, gb_predict(X_train, trees, eta)))
            test_errors.append(mean_squared_error(y_test, gb_predict(X_test, trees, eta)))

        trees.append(tree)
        
    return trees, train_errors, test_errors

def gb_fit_stochastic(n_trees, max_depth, X_train, X_test, y_train, y_test, eta, subsample):
    trees = []
    train_errors = []
    test_errors = []
    
    for i in range(n_trees):
        # Создаем решающее дерево с ограничением глубины
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        
        # Выбираем случайную подвыборку данных для обучения дерева
        indices = np.random.choice(len(X_train), size=int(subsample * len(X_train)), replace=False)
        X_train_subsample, y_train_subsample = X_train[indices], y_train[indices]

        if len(trees) == 0:
            # Обучаем первое дерево на случайной подвыборке
            tree.fit(X_train_subsample, y_train_subsample)
            
            # Вычисляем ошибку на обучающей и тестовой выборках
            train_errors.append(mean_squared_error(y_train, gb_predict(X_train, trees, eta)))
            test_errors.append(mean_squared_error(y_test, gb_predict(X_test, trees, eta)))
        else:
            # Получаем предсказания текущей композиции на случайной подвыборке
            target = gb_predict(X_train_subsample, trees, eta)
            
            # Обучаем дерево на остатках по случайной подвыборке
            tree.fit(X_train_subsample, residual(y_train_subsample, target))
            
            # Вычисляем ошибку на обучающей и тестовой выборках
            train_errors.append(mean_squared_error(y_train, gb_predict(X_train, trees, eta)))
            test_errors.append(mean_squared_error(y_test, gb_predict(X_test, trees, eta)))

        trees.append(tree)
        
    return trees, train_errors, test_errors


n_trees = 10

# Максимальная глубина деревьев
max_depth = 3

# Шаг
eta = 1

trees, train_errors, test_errors = gb_fit(n_trees, max_depth, X_train, X_test, y_train, y_test, eta)



def evaluate_alg(X_train, X_test, y_train, y_test, trees, eta):
    train_prediction = gb_predict(X_train, trees, eta)

    print(f'Ошибка алгоритма из {n_trees} деревьев глубиной {max_depth} \
    с шагом {eta} на тренировочной выборке: {mean_squared_error(y_train, train_prediction)}')

    test_prediction = gb_predict(X_test, trees, eta)

    print(f'Ошибка алгоритма из {n_trees} деревьев глубиной {max_depth} \
    с шагом {eta} на тестовой выборке: {mean_squared_error(y_test, test_prediction)}')

def get_error_plot(n_trees, train_err, test_err):
    plt.xlabel('Iteration number')
    plt.ylabel('MSE')
    plt.xlim(0, n_trees)
    plt.plot(list(range(n_trees)), train_err, label='train error')
    plt.plot(list(range(n_trees)), test_err, label='test error')
    plt.legend(loc='upper right')
    plt.show()


#evaluate_alg(X_train, X_test, y_train, y_test, trees, eta)
#get_error_plot(n_trees, train_errors, test_errors)


def number_of_tree_analyzing():
    print("Построение графика зависимости ошибки от размера леса")
    # Изменение количества деревьев
    n_trees_list = list(range(5, 70))
    max_depth = 2
    eta = 0.1
    
    train_errors_list = []
    test_errors_list = []
    
    for n_trees in n_trees_list:
        print("Обучение ", n_trees, "деревьев")
        trees, train_errors, test_errors = gb_fit(n_trees, max_depth, X_train, X_test, y_train, y_test, eta)
        train_errors_list.append(train_errors[-1])  # Используем ошибку на последней итерации
        test_errors_list.append(test_errors[-1])
        
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(n_trees_list, train_errors_list, label='train error')
    plt.plot(n_trees_list, test_errors_list, label='test error')
    plt.xlabel('Количество деревьев в лесу')
    plt.ylabel('MSE')
    plt.title('Ошибка в зависимости от деревьев в лесу')
    plt.legend()


def max_deph_analyzing():
    print("Построение графика зависимости ошибки от глубины деревьев")
    # Изменение максимальной глубины деревьев
    n_trees = 10
    max_depth_list = list(range(2, 10))
    eta = 0.1
    
    train_errors_list = []
    test_errors_list = []
    
    for max_d in max_depth_list:
        print("Обучение на глубине в ", max_d)
        trees, train_errors, test_errors = gb_fit(n_trees, max_d, X_train, X_test, y_train, y_test, eta)
        train_errors_list.append(train_errors[-1])  # Используем ошибку на последней итерации
        test_errors_list.append(test_errors[-1])
        
    plt.subplot(1, 2, 2)
    plt.plot(max_depth_list, train_errors_list, label='train error')
    plt.plot(max_depth_list, test_errors_list, label='test error')
    plt.xlabel('Max Depth of Trees')
    plt.ylabel('MSE')
    plt.title('Error vs Max Depth of Trees')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def stochastic_train_and_test_analyzing(): 
    print("Построение графика зависимости ошибки от количества итараций в стохастическом градиентном бустинге")
    # Параметры для стохастического градиентного бустинга
    n_trees = 25
    max_depth = 3
    eta = 0.1
    subsample = 0.5
    
    # Обучение стохастического градиентного бустинга
    trees_stochastic, train_errors_stochastic, test_errors_stochastic = gb_fit_stochastic(
        n_trees, max_depth, X_train, X_test, y_train, y_test, eta, subsample
    )
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    
    plt.plot(range(1, n_trees + 1), test_errors_stochastic, label='SGDBoost Test Error', linewidth=2)
    plt.plot(range(1, n_trees + 1), train_errors_stochastic, label='SGDBoost  Train Error', linewidth=2)
    
    plt.xlabel('Number of Trees')
    plt.ylabel('MSE')
    plt.title('Test Error for SGDBoost with Subsample Size = 0.5')
    plt.legend()
    plt.grid(True)
    plt.show()


#number_of_tree_analyzing()
#max_deph_analyzing()


#stochastic_train_and_test_analyzing()

