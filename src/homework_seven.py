import os

import numpy as np
import math as mt
from numpy.fft import rfft, irfft
import math
from random import random
from cmath import exp, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

abs_path = "/Users/evandonohue/PycharmProjects/Computational Physics/output"


def prob_decay(t, tau):
    return 1 - 2**(-t / tau)


def problem_one():
    print("Problem One")
    deltaT = 1  # s

    Bi_213 = 10000
    Pb_209 = 0
    Bi_209 = 0
    Tl_209 = 0

    tau_Bi_213 = 2760  # s
    tau_Pb_209 = 198  # s
    tau_Tl_209 = 132  # s

    P_Pb209 = prob_decay(deltaT, tau_Pb_209)
    P_Bi213 = prob_decay(deltaT, tau_Bi_213)
    P_Tl209 = prob_decay(deltaT, tau_Tl_209)

    Bi213 = []
    Pb209 = []
    Bi209 = []
    Tl209 = []

    t = 0
    Tpoints = []
    runTime = 20000  # s
    while t < runTime:
        Tpoints.append(t)
        Bi213.append(Bi_213)
        Pb209.append(Pb_209)
        Bi209.append(Bi_209)
        Tl209.append(Tl_209)
        for atom in range(0, Pb_209):
            if random() < P_Pb209:
                Pb_209 -= 1
                Bi_209 += 1
        for atom in range(0, Tl_209):
            if random() < P_Tl209:
                Tl_209 -= 1
                Pb_209 += 1
        for atom in range(0, Bi_213):
            if random() < P_Bi213:
                if random() < 0.9791:
                    Bi_213 -= 1
                    Pb_209 += 1
                else:
                    Bi_213 -= 1
                    Tl_209 += 1
        t += 1

    plt.plot(Tpoints, Bi213, label="Bi213")
    plt.plot(Tpoints, Pb209, label="Pb209")
    plt.plot(Tpoints, Bi209, label="Bi209")
    plt.plot(Tpoints, Tl209, label="Tl209")
    plt.legend()
    plt.show()


def func(x):
    return mt.cos(x) + mt.cos(mt.sqrt(2) * x) + mt.cos(mt.sqrt(3) * x)


def func_b(x):
    return x ** 2 - mt.cos(4 * pi * x)


def random_gauss():
    return float(np.random.normal(0, 1))  # Centered @ 0, std dev 1


def problem_two():
    print("Problem Two")
    t = 0
    Tmin = 0.001
    Tmax = 1.0
    Temp = 1.0
    x = 2
    tau = 1e4

    f = func(x)

    xs = []  # Xs we're trying
    Tpoints = []

    while (Tmin <= Temp):

        t += 1  # Increment here since we did the nought case outside the loop
        Temp = Tmax * mt.exp(-t / tau)

        temp = x
        temp_f = f
        x += random_gauss()
        f = func(x)

        # The exponential cooling schedule
        if random() > mt.exp(-(f - temp_f) / Temp):
            x = temp
            f = temp_f
        xs.append(x)
        Tpoints.append(t)

    plt.scatter(Tpoints, xs)
    plt.title("Simulated Annealing Part a")
    plt.show()

    # part b
    t = 0
    Tmin = 0.001
    Tmax = 25
    Temp = 1.0
    x = 40  # Close to the middle of the range
    tau = 1e4

    f = func_b(x)

    xs = []  # Xs we're trying
    Tpoints = []

    while (Tmin <= Temp):

        t += 1  # Increment here since we did the nought case outside the loop
        Temp = Tmax * mt.exp(-t / tau)

        temp = x
        temp_f = f
        x += random_gauss()
        f = func_b(x)

        # The exponential cooling schedule
        if random() > mt.exp(-(f - temp_f) / Temp):
            x = temp
            f = temp_f
        xs.append(x)
        Tpoints.append(t)

    plt.scatter(Tpoints, xs)
    plt.title("Simulated Annealing Part b")
    plt.show()


def problem_three():
    print("Problem Three")
    L = 101
    # Start at the middle
    x = 51
    y = 51
    t = 0
    # runTime = 1e6
    runTime = 1e3  # This gives more interesting results
    xs = []
    ys = []
    while t < runTime:
        prob = random()
        if prob < 0.25:
            if x < L:
                x += 1  # Move to the right
            else:
                t -= 1
        elif prob < 0.50:
            if x > 0:
                x -= 1  # Move to the left
            else:
                t -= 1
        elif prob < 0.75:
            if y > 0:
                y -= 1  # Move down
            else:
                t -= 1
        else:
            if y < L:
                y += 1  # Move up
            else:
                t -= 1
        t += 1
        # print(str(x) + ", " + str(y))
        print(t)
        xs.append(x)
        ys.append(y)

    plt.plot(xs, ys)
    plt.show()


def problem_four():
    print("Problem Four")
    L = 101
    # Start at the middle
    x = 51
    y = 51
    t = 0
    # runTime = 1e6
    runTime = 1e6  # This gives more interesting results
    xs = []
    ys = []

    anchored = {} # A dictionary of pairs of (x,y) values
    # The actual value won't matter. I'm just using the fact these have keys
    # I can easily check against to hold the spots

    while t < runTime:
        prob = random()
        if prob < 0.25:
            if x < L and ((x+1, y) not in anchored):
                x += 1  # Move to the right
            else:
                anchored[(x, y)] = True  # Assign this spot as taken
                print("Stuck @ " + str(x) + ", " + str(y))
                # Since that spot is now taken, start again from the middle
                x = 51
                y = 51
                t -= 1
        elif prob < 0.50:
            if x > 0 and ((x-1, y) not in anchored):
                x -= 1  # Move to the left
            else:
                anchored[(x, y)] = True
                print("Stuck @ " + str(x) + ", " + str(y))
                x = 51
                y = 51
                t -= 1
        elif prob < 0.75:
            if y > 0 and ((x, y-1) not in anchored):
                y -= 1  # Move down
            else:
                anchored[(x, y)] = True
                print("Stuck @ " + str(x) + ", " + str(y))
                x = 51
                y = 51
                t -= 1
        else:
            if y < L and ((x, y+1) not in anchored):
                y += 1  # Move up
            else:
                anchored[(x, y)] = True
                print("Stuck @ " + str(x) + ", " + str(y))
                x = 51
                y = 51
                t -= 1
        t += 1
        # print(str(x) + " " + str(y))

    # print(anchored)

    for val in anchored:
        xs.append(val[0])
        ys.append(val[1])

    plt.scatter(xs, ys, s=0.8, c='red')
    plt.show()


def run():
    print("\t\t\t\tHomework Set #7\n")
    # problem_one()
    # problem_two()
    # problem_three()
    problem_four()

    print("End of Homework #7")
