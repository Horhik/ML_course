import matplotlib.pyplot as plt

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

