import os

import numpy as np
from numpy.fft import rfft, irfft
import math
from cmath import exp, pi
import matplotlib.pyplot as plt

abs_path = "/Users/evandonohue/PycharmProjects/Computational Physics/output"


def square_wave(x):
    if x <= 1:
        return 1.0
    elif x > 1:
        return 0.0


def dft(y):
    N = len(y)
    c = np.zeros(N//2+1, complex)
    for k in range(N//2+1):
        for n in range(N):
            c[k] += y[n]*exp(-2j*pi*k*n / N)

    return c


def problem_one():
    print("Problem One")
    dummy_one = np.zeros(500, float)
    dummy_two = np.zeros(500, float)
    dummy_one.fill(1.0)
    dummy_two.fill(0.0)

    result = np.concatenate([dummy_one, dummy_two])
    plt.plot(result)
    plt.show()
    ck = dft(result)
    plt.plot(abs(ck))
    plt.show()

    N = 1000
    sawtooth = np.arange(N)
    ck2 = dft(sawtooth)
    plt.plot(abs(ck2))
    plt.show()

    dummy_three = np.arange(N)
    mod_sin = np.sin(pi * dummy_three / N) * np.sin(20 * pi * dummy_three / N)
    ck3 = dft(mod_sin)
    plt.plot(abs(ck3))
    plt.show()


def problem_two():
    print("Problem Two")
    os.chdir("/Users/evandonohue/PycharmProjects/Computational Physics/data/")
    month, spot_num = np.loadtxt("sunspots.txt", delimiter='\t', unpack=True)
    plt.plot(month, spot_num)
    plt.title("Sunspots Over Time")
    plt.show()
    # Looks like they repeat every ~110 months, so roughly 10 years
    ck = dft(spot_num)
    plt.plot(abs(ck)**2)
    plt.show()
    # Peak @ x ~= 24


def problem_three():
    print("Problem Three")
    os.chdir("/Users/evandonohue/PycharmProjects/Computational Physics/data/")
    prices = np.loadtxt("dow.txt", delimiter='\t', unpack=True)
    day = np.arange(len(prices))
    plt.plot(day, prices)
    plt.show()

    coeffs = rfft(prices)
    N = len(coeffs)
    print(N)
    coeffs[N//10:] = 0
    invft = irfft(coeffs)
    plt.plot(invft)
    plt.plot(prices)
    plt.title("First 10%")
    plt.show()

    coeffs = rfft(prices)
    n = round(len(coeffs) * 0.02)
    coeffs[n:] = 0
    invft = irfft(coeffs)
    plt.plot(prices)
    plt.plot(invft)
    plt.title("First 2%")
    plt.show()


def problem_four():
    print("Problem Four")
    N = 1000
    t = np.arange(0, 3)
    # t = np.linspace(0, 10, 100)

    for i in range(0, len(t)):
        num = t[i]
        # num = np.floor(t[i])
        # This produces a flat line otherwise...
        if num % 2 == 0:
            t[i] = 1
        else:
            t[i] = -1

    coeffs = rfft(t)
    coeffs[10:] = 0
    invf = irfft(coeffs)
    plt.plot(t)
    plt.plot(invf)
    plt.show()


def discrete_cos_trans(y):
    N = len(y)
    c = np.empty(2*N, float)

    for k in range(N):
        c[k] = y[k]
        c[2*N-k-1] = y[k]

    cfft = rfft(c)
    phi = exp(-1j * pi * np.arange(N)/(2*N))
    return np.real(cfft[:N] * phi)


def inverse_dct(y):
    N = len(y)
    c = np.zeros(N + 1, complex)
    phi = exp(1j * pi * np.arange(N) / (2 * N))
    c[:N] = y * phi
    result = irfft(c)
    return result


def problem_five():
    print("Problem Five")
    os.chdir("/Users/evandonohue/PycharmProjects/Computational Physics/data/")
    prices = np.loadtxt("dow2.txt", delimiter='\t', unpack=True)
    day = np.arange(len(prices))
    plt.plot(day, prices)
    plt.show()

    coeffs = rfft(prices)
    n = round(len(coeffs) * 0.02)
    coeffs[n:] = 0
    invft = irfft(coeffs)
    plt.plot(prices)
    plt.plot(invft)
    plt.title("First 2%")
    plt.show()

    coeffs = discrete_cos_trans(prices)
    n = round(len(coeffs) * 0.02)
    invft = inverse_dct(coeffs)
    plt.plot(invft)
    plt.plot(prices)
    plt.show()


def run():
    print("\t\t\t\tHomework Set #5\n")
    problem_one()
    # problem_two()
    # problem_three()
    # problem_four()
    # problem_five()

    print("End of Homework #5")
