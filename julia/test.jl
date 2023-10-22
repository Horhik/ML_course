using Pkg;
Pkg.add("Plots")
using Plots;
X = [1  1;
         1  1;
         1  2;
         1  5;
         1  3;
         1  0;
         1  5;
         1 10;
         1  1;
         1  2;];

y = [45, 55, 50, 55, 60, 35, 75, 80, 50, 60]

plot(X[:, 2], y, label="datasetw")

