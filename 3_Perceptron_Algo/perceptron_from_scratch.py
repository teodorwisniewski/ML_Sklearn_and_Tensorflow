import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random


def step_function(t):
    if t>=0:
        return 1
    return 0


def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])


def perceptron_one_epoch(X, y, W, b, learning_rate=0.01):
    for i in range(len(X)):
        x_values = (X.iloc[i, :]).values.reshape((1, 2))
        y_hat = step_function(np.dot(x_values, W) + b)
        if y.iloc[i, 0] - y_hat == 1:
            W = np.add(W, learning_rate * x_values.T)
            b += learning_rate
        elif y.iloc[i, 0] - y_hat == -1:
            W = np.add(W, -learning_rate * x_values.T)
            b -= learning_rate
    return W, b


def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b



df = pd.read_csv("data.csv", header=None)
X = df.iloc[:,0:2]
y = df.iloc[:,2:]
cond = (y==1).values

plt.scatter(X.iloc[cond,0], X.iloc[cond,1],label="positive")
plt.scatter(X.iloc[~cond,0], X.iloc[~cond,1],label="negative")
plt.legend()
plt.show


max_x1, min_x1 = max(X.iloc[:,0]), min(X.iloc[:,0])
max_x2, min_x2 = max(X.iloc[:,1]), min(X.iloc[:,1])
x1_vec = np.linspace(max_x1, min_x1, 100)
learning_rate = 0.1
W_vec = np.array(np.random.rand(2,1))
b = np.random.rand(1)[0]
W_vec, b, W_vec.shape

epochs = 100
line = -W_vec[0]/W_vec[1]*x1_vec-b/W_vec[1]
plt.plot(x1_vec,line, label = "line number = 000")
for i in range(epochs):
    W_vec, b = perceptron_one_epoch(X,y,W_vec,b)
    # W_vec, b = perceptronStep(X.values, y.values, W_vec, b)

    print(i,W_vec, b)


line = -W_vec[0]/W_vec[1]*x1_vec-b/W_vec[1]
plt.plot(x1_vec,line, label = "line number = last " + str(i+1))
plt.scatter(X.iloc[cond,0], X.iloc[cond,1],label="positive")
plt.scatter(X.iloc[~cond,0], X.iloc[~cond,1],label="negative")
plt.legend()
plt.show()