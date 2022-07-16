import os

import numpy as np
from numpy.fft import rfft, irfft
import math
from cmath import exp, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

abs_path = "/Users/evandonohue/PycharmProjects/Computational Physics/output"


def phi_xyz_lap(phi, x, y, z, a):
    # Where phi is the function we're interested in solving, and x/y/z are the location
    # a is the offset
    omega = 0.80

    # The combined Gauss-Seidel Over-relaxation method
    xa = phi[x+a, y, z] + phi[x-a, y, z]
    ya = phi[x, y+a, z] + phi[x, y-a, z]
    za = phi[x, y, z+a] + phi[x, y, z-a]
    value = ((omega + 1.0) / 6.0) * (xa + ya + za) - (omega * phi[x, y, z])
    # print(value)
    return value


def rho(x, y):
    if (20 <= x < 40) and (20 <= y <= 40):
        return 1.0
    elif (60 <= x < 80) and (60 <= y <= 80):
        return -1.0
    else:
        return 0


def rho_3d(x, y, z):
    if (20 <= x < 40) and (20 <= y <= 40) and (20 <= z <= 40):
        return 1.0
    elif (60 <= x < 80) and (60 <= y <= 80) and (60 <= z <= 80):
        return -1.0
    else:
        return 0


def phi_xy_pois(phi, x, y, a):
    # Where phi is the function we're interested in solving, and x/y/z are the location
    # a is the offset

    # Solving 2D Poisson's equation
    xa = phi[x+a, y] + phi[x-a, y]
    ya = phi[x, y+a] + phi[x, y-a]
    value = (1.0 / 4.0) * (xa + ya) + (a**2 / 4) * rho(x, y)
    # print(value)
    return value


def phi_xyz_pois(phi, x, y, z, a):
    # Where phi is the function we're interested in solving, and x/y/z are the location
    # a is the offset

    # Solving 3D Poisson's equation
    xa = phi[x + a, y, z] + phi[x - a, y, z]
    ya = phi[x, y + a, z] + phi[x, y - a, z]
    za = phi[x, y, z + a] + phi[x, y, z - a]
    value = (1.0 / 6.0) * (xa + ya + za) + (a**2 / 6) * rho_3d(x, y, z)
    return value


def problem_one():
    print("Problem One")
    sides = 1  # meter, side lengths
    cell_num = 100  # cells per side length
    V_top = 1  # Top voltage
    V_else = 0  # Elsewhere

    target = 1e-6
    delta = 1.0

    phi_result = np.zeros([cell_num+1, cell_num+1, cell_num+1],float)
    phi_prime = np.empty([cell_num + 1, cell_num + 1, cell_num + 1],float)
    # Set the entire top of the box to V_top
    phi_result[:, :, cell_num] = V_top

    while delta >= target:
        for i in range(cell_num + 1):
            for j in range(cell_num + 1):
                for k in range(cell_num + 1):
                    if i == 0 or i == cell_num or j == 0 or j == cell_num or k == 0 or k == cell_num:
                        phi_prime[i, j, k] = phi_result[i, j, k]
                    else:
                        phi_prime[i, j, k] = phi_xyz_lap(phi_result, i, j, k, 1)  # a = 1 cm, 0.01 m

        delta = abs((phi_result - phi_prime).max())
        phi_result, phi_prime = phi_prime, phi_result

    # plt.imshow(phi_result)
    # plt.gray()
    # plt.hot()
    # plt.show()


def problem_two():
    print("Problem Two a")
    cell_num = 100
    target = 1e-3
    phi_result = np.zeros([cell_num + 1, cell_num + 1], float)
    phi_prime = np.empty([cell_num + 1, cell_num + 1], float)

    delta = 1.0
    while delta > target:
        for i in range(cell_num + 1):
            for j in range(cell_num + 1):
                if i == 0 or i == cell_num or j == 0 or j == cell_num:
                    phi_prime[i, j] = phi_result[i, j]
                else:
                    phi_prime[i, j]= phi_xy_pois(phi_result, i, j, 1)  # a = 1 cm, 0.01 m
        delta = abs((phi_result - phi_prime).max())
        phi_result, phi_prime = phi_prime, phi_result
        # print(delta)  # Useful for debugging but slows the program
        break

    data = phi_result[:, :]
    x = np.arange(0, 101, 1)
    y = np.arange(0, 101, 1)

    plt.contourf(x, y, data)
    plt.title("2D Poisson's Equation Solution")
    plt.show()

    print("Problem Two b")
    sides = 1  # meter, side lengths
    cell_num = 100  # cells per side length

    target = 0.2  # This takes _a while_ to run, so I'm limiting this to 0.2
    delta = 1.0

    phi_result = np.zeros([cell_num + 1, cell_num + 1, cell_num + 1])
    phi_prime = np.empty([cell_num + 1, cell_num + 1, cell_num + 1])

    while delta >= target:
        for i in range(cell_num + 1):
            for j in range(cell_num + 1):
                for k in range(cell_num + 1):
                    if i == 0 or i == cell_num or j == 0 or j == cell_num or k == 0 or k == cell_num:
                        phi_prime[i, j, k] = phi_result[i, j, k]
                    else:
                        # This seems to be appreciably slower
                        # phi_prime[i, j, k] = phi_xyz_pois(phi_result, i, j, k, 1)  # a = 1 cm, 0.01 m
                        phi_prime[i, j, k] = 0.1666 * (phi_result[i + 1, j, k] + phi_result[i - 1, j, k] +
                                                       phi_result[i, j + 1, k] + phi_result[i, j - 1, k] +
                                                       phi_result[i, j, k + 1] + phi_result[i, j, k - 1] ) + \
                                             ((2 / 6) * rho_3d(i, j, k))

        delta = abs((phi_result - phi_prime).max())
        phi_result, phi_prime = phi_prime, phi_result
        # print(delta)  # Useful for debugging but slows the program
        # break

    data_1 = phi_result[:, :, 30]  # Right through the middle of the + box
    data_2 = phi_result[:, :, 70]  # The negative box
    data_3 = phi_result[:, :, 50]  # Middle of the entire box
    # The other views would be the same by symmetry
    x = np.arange(0, 101, 1)
    y = np.arange(0, 101, 1)

    plt.contourf(x, y, data_1)
    plt.title("3D Poisson's Equation Solution (+ Box Slice)")
    plt.show()
    plt.contourf(x, y, data_2)
    plt.title("3D Poisson's Equation Solution (- Box Slice)")
    plt.show()
    plt.contourf(x, y, data_3)
    plt.title("3D Poisson's Equation Solution (Middle Slice)")
    plt.show()


def problem_three():
    print("Problem Three")
    length = 0.20  # m
    width = 0.20  # m
    height = 0.05  # m
    cell_num = 100  # cells per side length
    h = 0.01
    D = 4.25e-6  # m^2 / s
    N = 100
    a = height / N
    c = h * (D / a**2)  # 'k' gets mistaken in the loop otherwise

    t = 0.0  # Starting time, s
    runtime = 20  # Runtime, s
    epsilon = h / 1000

    T_hot = 50  # deg C
    T_cold = 0  # deg C

    T_r = np.zeros([cell_num + 1, cell_num + 1, cell_num + 1])
    T_p = np.empty([cell_num + 1, cell_num + 1, cell_num + 1])

    T_r[:, :, cell_num] = T_hot
    T_r[:, :, 0] = T_cold

    t1 = 0.01
    t2 = 0.1
    t3 = 0.4
    t4 = 1.0
    t5 = 10.0

    while t <= runtime:
        for i in range(cell_num):
            for j in range(cell_num):
                for k in range(cell_num):
                    T_p[i, j, k] = T_r[i, j, k] + c * ( T_r[i + 1, j, k] + T_r[i - 1, j, k] +
                                                   T_r[i, j + 1, k] + T_r[i, j - 1, k] +
                                                   T_r[i, j, k + 1] + T_r[i, j, k - 1] - 6.0 * T_r[i, j, k])
        T_r, T_p = T_p, T_r
        t += h
        if(abs(t-t1) < epsilon):
            data_1 = T_r[:, 50, :]  # Slice through the middle
            # The other views would be the same by symmetry
            x = np.arange(0, 101, 1)
            y = np.arange(0, 101, 1)

            plt.contourf(x, y, data_1)
            plt.title("Heat Flow t="+str(t1))
            plt.show()
        if (abs(t - t2) < epsilon):
            data_1 = T_r[:, 50, :]
            # The other views would be the same by symmetry
            x = np.arange(0, 101, 1)
            y = np.arange(0, 101, 1)

            plt.contourf(x, y, data_1)
            plt.title("Heat Flow t="+str(t2))
            plt.show()
        if (abs(t - t3) < epsilon):
            data_1 = T_r[:, 50, :]
            # The other views would be the same by symmetry
            x = np.arange(0, 101, 1)
            y = np.arange(0, 101, 1)

            plt.contourf(x, y, data_1)
            plt.title("Heat Flow t="+str(t3))
            plt.show()
        if (abs(t - t4) < epsilon):
            data_1 = T_r[:, 50, :]
            # The other views would be the same by symmetry
            x = np.arange(0, 101, 1)
            y = np.arange(0, 101, 1)

            plt.contourf(x, y, data_1)
            plt.title("Heat Flow t="+str(t4))
            plt.show()
        if (abs(t - t5) < epsilon):
            data_1 = T_r[:, :, 50]
            # The other views would be the same by symmetry
            x = np.arange(0, 101, 1)
            y = np.arange(0, 101, 1)

            plt.contourf(x, y, data_1)
            plt.title("Heat Flow t="+str(t5))
            plt.show()


    data_1 = T_r[:, :, 50]  # Right through the middle of the + box
    # The other views would be the same by symmetry
    x = np.arange(0, 101, 1)
    y = np.arange(0, 101, 1)

    plt.contourf(x, y, data_1)
    plt.title("Heat Flow t=20")
    plt.show()


def run():
    print("\t\t\t\tHomework Set #6\n")
    # problem_one()
    # problem_two()
    problem_three()

    print("End of Homework #6")
