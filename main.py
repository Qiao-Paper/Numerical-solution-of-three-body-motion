from scipy.integrate import solve_ivp
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define Parameters
G = 1  # gravitation constant
m1, m2, m3 = 1, 1, 1

v1x0, v1y0 = 0, 0
v2x0, v2y0 = 0, 0
v3x0, v3y0 = 0, 0  # initial speed

t_span = (0, 10)  # initial time and end time

for i in range(1, 2):
    x10, y10 = 1, 0
    x20, y20 = -2, i*3
    x30, y30 = 10, 6  # initial position
    init0 = (x10, y10, x20, y20, x30, y30, v1x0, v1y0, v2x0, v2y0, v3x0, v3y0)  # initial state


    def odefun(t, z, G, m1, m2, m3):
        x1, y1, x2, y2, x3, y3, v1x, v1y, v2x, v2y, v3x, v3y = z
        return [v1x, v1y, v2x, v2y, v3x, v3y,
                G * m2 * (x2 - x1) / (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (3 / 2)) + G * m3 * (x3 - x1) / (
                            ((x3 - x1) ** 2 + (y3 - y1) ** 2) ** (3 / 2)),
                G * m2 * (y2 - y1) / (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (3 / 2)) + G * m3 * (y3 - y1) / (
                            ((x3 - x1) ** 2 + (y3 - y1) ** 2) ** (3 / 2)),
                G * m1 * (x1 - x2) / ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (3 / 2) + G * m3 * (x3 - x2) / (
                            ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** (3 / 2)),
                G * m1 * (y1 - y2) / ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (3 / 2) + G * m3 * (y3 - y2) / (
                            ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** (3 / 2)),
                G * m1 * (x1 - x3) / (((x1 - x3) ** 2 + (y1 - y3) ** 2) ** (3 / 2)) + G * m2 * (x2 - x3) / (
                            ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** (3 / 2)),
                G * m1 * (y1 - y3) / (((x1 - x3) ** 2 + (y1 - y3) ** 2) ** (3 / 2)) + G * m2 * (y2 - y3) / (
                            ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** (3 / 2))]


    # input form of ode function
    track1 = solve_ivp(odefun, t_span, init0,  first_step=0.001, max_step=0.001, method='DOP853', dense_output=True,
                       args=(G, m1, m2, m3))
    print(track1)
    plt.scatter(track1.y[0], track1.y[1], s=1)
    plt.scatter(track1.y[2], track1.y[3], s=1)
    plt.scatter(track1.y[4], track1.y[5], s=1)
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.title('X values versus Y values')
    plt.show()
