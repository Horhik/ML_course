import matplotlib.pyplot as plt
import random

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from decision_tree import build_tree, predict, accuracy_metric,get_meshgrid, classify_object


import numpy as np
np.random.seed(42)

def gendata(samples=500):
    a,b = make_classification(n_samples=samples,
                                                                 n_features=2, n_informative=2, 
                                                                 n_classes=2, n_redundant=0, 
                                                                 n_clusters_per_class=1, random_state=13)
    return (a,b) 


def get_bootstrap(data, labels, N):
    n_samples = data.shape[0] 
    bootstrap = []
    
    for i in range(N):
        
        sample_index = np.random.randint(0, n_samples, size=n_samples)
        b_data = data[sample_index]
        b_labels = labels[sample_index]
        
        bootstrap.append((b_data, b_labels))
        
    return bootstrap

def get_subsample(len_sample):
    # будем сохранять не сами признаки, а их индексы
    sample_indexes = list(range(len_sample))

    len_subsample = int(np.round(np.sqrt(len_sample)))
    
    subsample = np.random.choice(sample_indexes, size=len_subsample, replace=False)


    return subsample


def not_in_bootstrap(data, tree, labels):
    notin = []
    notin_labels = []
    for i in range(data.shape[0]):
        #print(tree.bootstraped_data, data[i])
        if not data[i] in tree.bootstraped_data:
            notin.append(data[i])
            notin_labels.append(labels[i])
    return (notin, notin_labels)

def random_forest(data, labels, n_trees):
    forest = []
    bootstrap = get_bootstrap(data, labels, n_trees)

    
    for b_data, b_labels in bootstrap: 
        tree = build_tree(b_data, b_labels)
        # pushing list of bootstraped data into tree
        tree.bootstraped_data = b_data
        forest.append(tree)
        
    return forest


def tree_vote(forest, data):

    # добавим предсказания всех деревьев в список
    predictions = []
    for tree in forest:
        predictions.append(predict(data, tree))
    #print(predictions)

    # сформируем список с предсказаниями для каждого объекта
    predictions_per_object = list(zip(*predictions))
    #print(predictions_per_object)

    # выберем в качестве итогового предсказания для каждого объекта то,
    # за которое проголосовало большинство деревьев
    voted_predictions = []
    for obj in predictions_per_object:
        voted_predictions.append(max(set(obj), key=obj.count))
        
    return voted_predictions

def oob(tree, data, labels):
    fail_count = 0
    (not_bootstraped_data, not_bootstraped_labels) = not_in_bootstrap(data, tree, labels)
    N = len(not_bootstraped_labels) 
    for i in range(N):
        if classify_object(not_bootstraped_data[i], tree) != not_bootstraped_labels[i]:
            fail_count +=1
    tree.oob = fail_count/N
    return(fail_count/N)
    
        


classification_data, classification_labels = gendata()
colors = ListedColormap(['red', 'blue'])
light_colors = ListedColormap(['lightcoral', 'lightblue'])

plt.figure(figsize=(8,8))
#print(classification_labels)
plt.scatter(classification_data[:, 0], classification_data[:, 1], 
              c=classification_labels, cmap=colors);
#plt.show()


train_data, test_data, train_labels, test_labels = train_test_split(classification_data, 
                                                                    classification_labels, 
                                                                    test_size=0.3,
                                                                    random_state=1)


def train(number_of_trees=[1, 5, 10, 50, 100], colors_list=[['lightcoral', 'lightblue'], ['brown', 'azure'],  ['tomato', 'cyan'] , ['lightsalmon', 'skyblue'],  ['rosybrown', 'aqua']]):
    

    for i in range(len(number_of_trees)):
        print("Training on forest of ", number_of_trees[i], " trees ")
        colors = ListedColormap(['red', 'blue'])
        light_colors = ListedColormap(colors_list[i])
        n_trees = number_of_trees[i]
        forest = random_forest(train_data, train_labels, n_trees)
        
        train_answers = tree_vote(forest, train_data)   
        test_answers = tree_vote(forest, test_data)
        n = 0
        out_of_bag_error = 0
        for tree in forest:
            n+=1
            out_of_bag_error += oob(tree, train_data, train_labels)
        print("OOB is ", 1/n * out_of_bag_error)

            
        
        train_accuracy = accuracy_metric(train_labels, train_answers)
        print(f'Точность случайного леса из {n_trees} деревьев на обучающей выборке: {train_accuracy:.3f}')
        
        # Точность на тестовой выборке
        test_accuracy = accuracy_metric(test_labels, test_answers)
        print(f'Точность случайного леса из {n_trees} деревьев на тестовой выборке: {test_accuracy:.3f}')


        # график обучающей выборки
        xx, yy = get_meshgrid(train_data)
        data = np.c_[xx.ravel(), yy.ravel()]
        mesh_predictions = np.array(tree_vote(forest, data)).reshape(xx.shape)
        plt.pcolormesh(xx, yy, mesh_predictions, cmap = light_colors)
        plt.scatter(train_data[:, 0], train_data[:, 1], c = train_labels, cmap = colors)
        plt.title(f'Train accuracy={train_accuracy:.2f}, number of trees ={n_trees} ')
        plt.show()

        xx, yy = get_meshgrid(test_data)
        data = np.c_[xx.ravel(), yy.ravel()]
        mesh_predictions = np.array(tree_vote(forest, data)).reshape(xx.shape)
        plt.pcolormesh(xx, yy, mesh_predictions, cmap = light_colors)
        plt.scatter(test_data[:, 0], test_data[:, 1], c = test_labels, cmap = colors )
        plt.title(f'Test accuracy={test_accuracy:.2f}, number of trees ={n_trees} ')
        plt.show()

'''

        # график тестовой выборки
        plt.subplot(1,2,1)
        xx, yy = get_meshgrid(test_data)
        data = np.c_[xx.ravel(), yy.ravel()]
        mesh_predictions = np.array(tree_vote(forest, data)).reshape(xx.shape)
        plt.pcolormesh(xx, yy, mesh_predictions, cmap = light_colors)
        plt.scatter(test_data[:, 0], test_data[:, 1], c = test_labels, cmap = colors)
        plt.title(f'Test accuracy={test_accuracy:.2f}')
        '''

train()



#TODO Сделать выводы о получаемой сложности гиперплоскости и недообучении или переобучении случайного леса в зависимости от количества деревьев в нем.

