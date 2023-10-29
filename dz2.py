import matplotlib.pyplot as plt
import numpy as np

from errors import calc_mse
from gsd import stohastic_gsd

# Initializing dataset

X = np.array([[   1,    1,  500,    1],
              [   1,    1,  700,    1],
              [   1,    2,  750,    2],
              [   1,    5,  600,    1],
              [   1,    3, 1450,    2],
              [   1,    0,  800,    1],
              [   1,    5, 1500,    3],
              [   1,   10, 2000,    3],
              [   1,    1,  450,    1],
              [   1,    2, 1000,    2]])

y = [45, 55, 50, 55, 60, 35, 75, 80, 50, 60]

# defining resizing functions

def normalize(X):
    return (X - X.min()) / (X.max() - X.min())

def standartize(X):
    l = len(X)
    mu = 1/l * sum(X)
    sig = np.sqrt(1/l * sum(X - mu)**2)


# Normalizing X

space=np.arange(len(X[:, 2]))
X_norm = X.copy().astype(np.float64)

colors = ["red","green","blue","yellow","pinkd"]
for i in range(1, X.shape[1]):
    X_norm[:, i] = normalize(X_norm[:, i])
    

# drawing changes of normalization

fig, ((s1, s2, s3), (n1,n2,n3)) = plt.subplots(2, 3)
s1.hist(X[:,1], label="initial", color=colors[1])
s2.hist(X[:,2], label="initial", color=colors[2])
s3.hist(X[:,3], label="initial", color=colors[3])

n1.hist(X_norm[:,1], label="initial", color=colors[1])
n2.hist(X_norm[:,2], label="initial", color=colors[2])
n3.hist(X_norm[:,3], label="initial", color=colors[3])

for ax in fig.get_axes():
    ax.label_outer()

#plt.show()

#stohastic_gsd(X, y)
