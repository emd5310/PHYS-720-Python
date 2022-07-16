import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

abs_path = "/Users/evandonohue/PycharmProjects/Computational Physics/output"

"""
Interesting constants:
10, 60, 8/3

"""

# Lorenz constants
sigma = 10
r = 28
b = 8 / 3


def r_3(vec, t):
    dx = sigma * (vec[1] - vec[0])
    dy = r * vec[0] - vec[1] - vec[0] * vec[2]
    dz = vec[0] * vec[1] - b * vec[2]
    return np.array([dx, dy, dz])


# Initial Value
omega = 1


# vec[d_x, beta]
def sho(vec, t):
    d_x = vec[1]
    d_beta = -(omega ** 2) * vec[0]  # Parametrized as beta
    return np.array([d_x, d_beta])


# vec[d_x, beta]
def sho_verlet(vec):
    d_x = vec[1]
    d_beta = -(omega ** 2) * vec[0]  # Parametrized as beta
    return np.array([d_x, d_beta])


# vec[d_x, beta]
def sho_verlet_2(pos, vel):
    d_x = vel
    d_beta = -(omega ** 2) * pos  # Parametrized as beta
    return d_beta


def fourth_order_rk(func, init_vec, start, stop, steps):
    """
    Fourth Order Runge-Kutta Method - Fifth Order Error
    :param func: Function to solve, must return NumPy array of form [x,y,z]
    :param init_vec: Starting values for [x,y,z]
    :param start: Starting time
    :param stop: Stopping time
    :param steps: Number of steps between start,stop
    :return: A NumPy array containing all x,y,z points
    """
    h = (stop - start) / steps
    t_points = np.arange(start, stop, h)
    x_points = []
    y_points = []
    z_points = []

    r = np.array([init_vec[0], init_vec[1], init_vec[2]], float)

    for t in t_points:
        x_points.append(r[0])
        y_points.append(r[1])
        z_points.append(r[2])
        k1 = h * func(r, t)
        k2 = h * func(r + 0.5 * k1, t + 0.5 * h)
        k3 = h * func(r + 0.5 * k2, t + 0.5 * h)
        k4 = h * func(r + k3, t + h)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    results = np.array([x_points, y_points, z_points])

    return t_points, results


def second_order_rk(func, init_vec, start, stop, steps):
    """
    Second Order Runge-Kutta Method - Third Order Error
    :param func: Function to solve, must return NumPy array of form [x,y,z]
    :param init_vec: Starting values for [x,y,z]
    :param start: Starting time
    :param stop: Stopping time
    :param steps: Number of steps between start,stop
    :return: A NumPy array containing all x,y,z points
    """
    h = (stop - start) / steps
    t_points = np.arange(start, stop, h)
    x_points = []
    y_points = []
    z_points = []

    r = np.array([init_vec[0], init_vec[1], init_vec[2]], float)

    for t in t_points:
        x_points.append(r[0])
        y_points.append(r[1])
        z_points.append(r[2])
        k1 = h * func(r, t)
        k2 = h * func(r + 0.5 * k1, t + 0.5 * h)
        r += k2

    results = np.array([x_points, y_points, z_points])

    return t_points, results


def leapfrog(func, init_vec, start, stop, steps):
    """
    Leapfrog Method - Third Order Error
    :param func: Function to solve, must return NumPy array of form [x,y,z]
    :param init_vec: Starting values for [x,y,z]
    :param start: Starting time
    :param stop: Stopping time
    :param steps: Number of steps between start,stop
    :return: A NumPy array containing all x,y,z points
    """
    h = (stop - start) / steps
    t_points = np.arange(start, stop, h)
    x_points = []
    y_points = []

    r = np.array([init_vec[0], init_vec[1]])
    half_step = r + 0.5 * h * func(r)

    for t in t_points:
        x_points.append(r[0])
        y_points.append(r[1])

        r += h * func(half_step)
        half_step += h * func(r)

    results = np.array([x_points, y_points])

    return t_points, results


def verlet(func, init_pos, init_vel, start, stop, steps):
    h = (stop - start) / steps
    t_points = np.arange(start, stop, h)
    x_points = []

    r = np.array(init_pos)
    v = np.array(init_vel)
    vhalf = v + 0.5 * h * func(r, v)

    for t in t_points:
        x_points.append(r)

        r += h * vhalf
        k1 = h * func(r, v)
        v = vhalf + 0.5 * k1
        vhalf += k1

    results = x_points

    return t_points, results


def problem_one():
    print("Problem #1 - Lorenz Equations with SORK and FORK")
    start = 0
    stop = 50
    N = 20000
    init_vals = np.array([0.0, 1.0, 0.0], float)

    time, result_vec = second_order_rk(r_3, init_vals, start, stop, N)

    plt.plot(time, result_vec[1])
    plt.show()
    plt.plot(result_vec[0], result_vec[2])
    plt.show()


def problem_two():
    print("Problem #1 - Lorenz Equations with SORK and FORK")
    start = 0
    stop = 50
    N = 20000
    init_vals = np.array([0.0, 1.0, 0.0], float)

    time, result_vec = fourth_order_rk(r_3, init_vals, start, stop, N)

    plt.plot(time, result_vec[1])
    plt.show()
    plt.plot(result_vec[0], result_vec[2])
    plt.show()


def problem_four_a():
    start = 0
    stop = 50
    N = 20000

    x = 1
    dx_dt = 0
    init_vals = np.array([x, dx_dt], float)  # Specifying our ICs

    time, result_vec = leapfrog(sho_verlet, init_vals, start, stop, N)  # Calculate positions over the interval
    plt.plot(time, result_vec[1])
    plt.ylabel("Position")
    plt.xlabel("Time (s)")
    plt.title("SHO with LF")
    plt.grid()
    plt.show()


def problem_four_b():
    start = 0
    stop = 50
    N = 20000

    x = 2
    dx_dt = 0
    init_vals = np.array([x, dx_dt], float)  # Specifying our ICs

    time, result_vec = leapfrog(sho_verlet, init_vals, start, stop, N)  # Calculate positions over the interval
    plt.plot(time, result_vec[1])
    plt.ylabel("Position")
    plt.xlabel("Time (s)")
    plt.title("SHO with LF")
    plt.grid()
    plt.show()


def problem_five_a():
    start = 0
    stop = 50
    N = 20000

    x = 1
    init_pos = np.array(x, float)  # Specifying our ICs
    init_vel = np.array(0, float)  # Specifying our ICs

    time, result_vec = verlet(sho_verlet_2, init_pos, init_vel, start, stop, N)  # Calculate positions over the interval
    plt.plot(time, result_vec)
    plt.ylabel("Position")
    plt.xlabel("Time (s)")
    plt.title("SHO with Verlet")
    plt.grid()
    plt.show()


def problem_five_b():
    start = 0
    stop = 50
    N = 20000

    x = 2
    dx_dt = 0
    init_pos = np.array([x, dx_dt], float)  # Specifying our ICs
    init_vel = np.array([0, 0], float)  # Specifying our ICs

    time, result_vec = verlet(sho_verlet, init_pos, init_vel, start, stop, N)  # Calculate positions over the interval
    plt.plot(time, result_vec[1])
    plt.ylabel("Position")
    plt.xlabel("Time (s)")
    plt.title("SHO with Verlet")
    plt.grid()
    plt.show()


def run():
    print("\t\t\t\tHomework Set #4\n")
    problem_one()
    problem_two()
    problem_four_a()
    problem_four_b()
    problem_five_a()

    print("End of Homework #4")
