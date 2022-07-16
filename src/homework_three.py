import numpy as np
import math
import matplotlib.pyplot as pyp
from util import plotting as plt
from util.gaussxw import gaussxw


def gaussian_quad(func, N, a, b):
    x, w = gaussxw(50)
    points = 0.5 * (b - a) * x + 0.5 * (b + a)
    weights = 0.5 * (b - a) * w

    result = 0
    for i in range(N):
        result += weights[i] * func(points[i])
    return result


def integral(x):
    return (x**4 * np.exp(x))/((np.exp(x) - 1)**2)


def problem_three():
    # Part a
    V = 0.001  # m^3
    rho = 6.022e28
    thetad = 428
    k_b = 1.38e-23
    T = 274  # Kelvin, about room temp
    a = 0
    b = thetad / T
    N = 50
    print(9 * V * rho * k_b * (T / thetad)**3 * gaussian_quad(integral, N, a, b))

    # Part b
    temps = np.linspace(5, 500, 500)
    results = []
    for temp in temps:
        b = thetad / temp
        results.append(9 * V * rho * k_b * (temp / thetad) ** 3 * gaussian_quad(integral, N, a, b))

    # print(results)
    plt.plot(temps, results, "Heat Capcity", "Temp. (K)", "Capacity", "CompHomework3_3", "CompHomework3_3", False,
             False)


def prob_four_func(x):
    return 1 + (1.0/2.0) * np.tanh(2 * x)


def exact_formula(x):
    return 1 / (np.cosh(2 * x) ** 2)


def central_difference(func, a, b, N):
    h = (b - a) / N
    x = a
    deriv = []

    for i in range(N):
        deriv.append((func(x + h) - func(x - h)) / (2 * h))
        x += h

    return deriv


def problem_four():
    x_points = np.linspace(-2, 2, 75)
    deriv = central_difference(prob_four_func, -2, 2, 75)
    exact_deriv = exact_formula(x_points)

    pyp.plot(x_points, deriv, label="Numerical", marker="x")
    pyp.plot(x_points, exact_deriv, label="Exact")
    pyp.title("Derivative of prob_four_func")
    pyp.ylabel("y")
    pyp.xlabel("x")
    pyp.legend()
    pyp.show()
    # Even 75 points across this interval gives very good agreement to the exact formula


def run():
    print("\t\t\t\tHomework Set #3 - Problems 3 & 4\n")
    problem_three()
    problem_four()
    print("")
