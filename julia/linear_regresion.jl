#import Pkg;
#Pkg.add("Plots")
#Pkg.add("PlotThemes")
using Plots
using PlotThemes
using Statistics
using Printf
using LinearAlgebra


theme(:orange)
X = [1  1;
         1  1;
         1  2;
         1  5;
         1  3;
         1  0;
         1  5;
         1 10;
         1  1;
     1  2;
     1  4;
     1  8;

     ];

y = [45, 55, 50, 55, 60, 35, 75, 80, 50, 60, 70, 75]


y_pred1 = 5 * X[:, 2] + 35 * X[:, 1] 
y_pred2 = 7.5 * X[:, 2] + 40 * X[:, 1]

#plot()
#scatter!(X[:, 2], y, label="dataset")

c_err(y0, y) = sum(y0 - y)
c_mae(y0, y) = length(y)*mean(abs.(y0 - y))
c_mse(y0, y) = (1/length(y))*sum((y0 - y).^2)


function c_all(y0, y)
    @printf("Error: %f \n", c_err(y0,y))
    @printf("Mean Absolut Error: %f \n", c_mae(y0,y))
    @printf("Mean Squared Error: %f \n", c_mse(y0,y))
    end

𝑤=(X'*X)^(-1)*X'*y
print(𝑤, "\n")

w_pred = 𝑤[1] * X[:, 1] + 𝑤[2] * X[:, 2]

#p_lin = plot(X[:, 2], [y_pred1  y_pred2 w_pred], label=["prediction 1" "prediction 2" "analytical solution"])

c_all(y, y_pred1)
c_all(y, y_pred2)
c_all(y, w_pred)



#scatter!(X[:, 2], y, label="dataset")

# Counting Gradient Discease


# Random weigts
#print("Weights are: ", W, "\n")
# Estimated Time of Arrival
global eta = 0.001
# Size of dataset
n = length(y)
# Differential of Q
diffQ(W_) = 2/n*X'*(X*W_ - y)
move_down(W_o, W_) = W_o - eta*diffQ(W_)  

# Moving backwards gradient
global W = rand(size(X[:, :])[2])


md(we)=move_down(W, we)
progress = [X*W]
err_prog = []
mae_prog = []
mse_prog = []


#plot(X[:, 2], X*W, label="Gradient descent", color=("#"*string(3*111111)))
#global c = 1;
#eps = 10.01
# while c_mse(y, X*W) > eps 
#     W_ = W
#     global W = move_down(W_, W_)
#     print("weigths are: ", W, "\n")
#     c_all(y, X*W)
#     push!(progress, X*W)

#     push!(err_prog, c_err(y, X*W))
#     push!(mae_prog, c_mae(y, X*W))
#     push!(mse_prog, c_mse(y, X*W))
    
#     #global c+=1
#     #global eta/=c
#     end
function ∂Q(Y::Array, X::Matrix, ω::Array)
    return 2/length(Y)*X'*(X*ω - Y)
    end

#w1 = range(-500, 500, 5000)
#w2 = range(-500, 500, 5000)

#QQ(w1,w2)= c_mse(y, X*∂Q(y, X, [w1,w2]))
#print(" KEKEK ", QQ, " KEKKEK \n")

function ↓(X, Y, ∂Q::Function, eta, ψ, ε)
    w = rand(2)
    w_change = ε*10
    errors = []
    weights = []
    i = 1
    error = ε*10
    while w_change > ε && i < ψ
        Q = ∂Q(Y,X,w)
        next_w = w - eta*Q
        error = c_mse(y, X*next_w)

        w_change = norm(next_w-w)

        push!(errors, error)
        push!(weights, next_w)
        print("error for ", w_change, " is ", error, "\n")

        i+=1
        w = next_w
        eta/=i
    end

    return (weights, errors, length(weights))

    end
op = []
print(op)



tries = 100
(weights,errors,sz)  = ↓(X, y, ∂Q, 0.08, tries+1, 0.001)
tries=sz
sols =  map(x -> X*x, weights)

 print(length(weights), "\n")
 print(length(errors), "\n")
 print("drawing grapigcs")

 weight = scatter(X[:, 2], y, label="Dataset")
 for i in 1:tries
     plot!(X[:, 2], sols, label="From Gradient Descent",  legend=false, opacity=((10*i/tries)), color=:green)
     end

# plot!(X[:, 2], X*W, label="Gradient descent", color=("#"*string(3*111111)))
# plot!(X[:, 2], w_pred, label="Analytical", color=:red)

# stat = plot(X[:, 2], err_prog, label="Error", color=:red, linestyle=:dash)
# plot!(X[:, 2], mae_prog, label="Mean Abs", color=:pink, linestyle=:dash)
# plot!(X[:, 2], mse_prog, label="Mean square", color=:rose, linestyle=:dash)

plot(weight)

#print("printing plot: \n" )


#svg("grad.png")

#surface(w1,w2,QQ)
