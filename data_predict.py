import random
import math
import matplotlib.pyplot as plt
import numpy as np
import csv
from neural_net_train import Neural_net

fig = plt.figure()
random.seed(69)

def get_neighbors(Tn, i, j):
    T_west = Tn[i][j-1]
    T_east = Tn[i][j+1]
    T_north = Tn[i-1][j]
    T_south = Tn[i+1][j]
    T_center = Tn[i][j]

    return T_west, T_north, T_east, T_south, T_center

def get_Tn_1(T, i, j):
    return round(T[i][j], 3)

N = 5
L = 1
t = 0
t_end = 2e-2
dx = L/N
dy = L/N
alfa = 100
dt = 1/4 * dx**2 / alfa * 1 / 5

x = np.linspace(-L/2, L/2, N)
y = x
x, y = np.meshgrid(x, y)


T = [[random.random() for i in range(N)] for i in range(N)]

nn = Neural_net(2, (16, 16), 1)
nn.load_model('he_2(16, 16)1')

#OKRAJOVÉ PODMÍNKY
for i in range(N):
    T[0][i] = 0
    T[N - 1][i] = 1
    T[i][0] = 1
    T[i][N - 1] = 1


while t < t_end:
    #SOLVE
    plt.contourf(x, y, T)
    plt.colorbar()
    plt.scatter(x, y, s=1)
    # plt.imshow(T, cmap='gray', interpolation='nearest')
    plt.draw()
    plt.pause(0.1)
    fig.clear()
    t += dt
    print(t*10e6)

    Tn = np.copy(T)
    for i in range(1, N-1):
        for j in range(1, N-1):   
            T_center = Tn[i][j]
            T_4_sum = Tn[i+1][j] + Tn[i-1][j] + Tn[i][j+1] + Tn[i][j-1]
            T_new = nn.predict([[T_center, T_4_sum]])
            #if T_new >= 0 and T_new <= 1:
            T[i][j] = T_new[0][0] * 1
            # elif T_new < 0:
            #      T[i][j] = 0.
            # elif T_new > 1:
            #     T[i][j] = 1.

        print(i) 




plt.contourf(x, y, T)
plt.colorbar()
plt.scatter(x, y, s=1)
plt.show()
