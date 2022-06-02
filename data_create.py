import random
import math
import matplotlib.pyplot as plt
import numpy as np
import csv

from sympy import numbered_symbols

fig = plt.figure()
random.seed(8)

def get_neighbors(Tn, i, j):
    T_west = round(Tn[i][j-1], 3)
    T_east = round(Tn[i][j+1], 3)
    T_north = round(Tn[i-1][j], 3)
    T_south = round(Tn[i+1][j], 3)
    T_center = round(Tn[i][j], 3)

    return T_west, T_north, T_east, T_south, T_center

def get_Tn_1(T, i, j):
    return round(T[i][j], 3)

N = 10
L = 1
t = 0
t_end = 4e-3
dx = L/N
dy = L/N
alfa = 100
dt = 1/4 * dx**2 / alfa

x = np.linspace(-L/2, L/2, N)
y = x
x, y = np.meshgrid(x, y)


T = [[0 for i in range(N)] for i in range(N)]

#OKRAJOVÉ PODMÍNKY
for i in range(N):
    T[0][i] = 0
    T[N - 1][i] = 1
    T[i][0] = 1
    T[i][N - 1] = 1
data_points = 25
with open('input.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for t_4_sum_i in range(data_points):
        for t_center_i in range(data_points):

            T_4_sum = round(t_4_sum_i / data_points * 4, 4) #round(random.uniform(0, 4), 5)
            T_center = round(t_center_i / data_points, 4) #round(random.uniform(0, 1), 5)

            T_new = T_center + 0.1 *(T_4_sum - 4 * T_center)
            print(T_new)
            T_new = round(T_new, 10)


            row = (T_center, T_4_sum, T_new)
            print(row)
            writer.writerow(row)

        