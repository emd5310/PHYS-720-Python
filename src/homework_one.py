import util.plotting as plt
import numpy as np
import matplotlib.pyplot as pyp
import math

# Desired tolerance for problem #1 and #2
EPSILON_1 = 1e-3
EPSILON_2 = 1e-10


def bisection_method(func, x_1, x_2, tolerance, verbose=False):
    # Check that they are of opposite sign
    if np.sign(func(x_1)) != np.sign(func(x_2)):
        if verbose:
            print("Sign check passed! This method will work.")
        f_x_1 = func(x_1)  # Calculate this now so we don't do this check on every loop iteration

        # As long as we aren't within the given tolerance, keep iterating
        while abs(x_1 - x_2) >= tolerance:
            x_prime = (1 / 2) * (x_1 + x_2)  # Calculate the new midpoint
            f_x_prime = func(x_prime)  # Save the result of the function at that point

            # Change one of our bracketing points based on which share a sign with x_prime
            if np.sign(f_x_prime) == np.sign(f_x_1):
                x_1 = x_prime
            else:  # must be x_2
                x_2 = x_prime

        # The loop breaks once we've reached our desired tolerance, so we return the final value
        return (1/2) * (x_1 + x_2)

    else:
        if verbose:
            print("Bisection method condition not met- initial points must have opposite sign!")
        return math.nan


# ODD STATES
def func_a(E):
    # Define our constants...
    w = 1.0e-9  # nm
    m_e = 9.1094e-31  # kg
    e = 1.602e-19  # C
    hbar = 1.0546e-34
    V = 20  # eV

    return np.tan(np.sqrt((w ** 2 * m_e * e * E) / (2 * hbar ** 2))) + np.sqrt(E / (V - E))


# EVEN STATES
def func_b(E):
    # Define our constants...
    w = 1.0e-9  # nm
    m_e = 9.1094e-31  # kg
    e = 1.602e-19  # C
    hbar = 1.0546e-34
    V = 20  # eV

    return np.tan(np.sqrt((w ** 2 * m_e * e * E) / (2 * hbar ** 2))) - np.sqrt((V-E) / E)


def problem_one():
    print("Running problem #1...\n")
    E_i = 0.0001  # eV
    E_f = 20.0001  # eV
    w = 1.0e-9  # nm
    m_e = 9.1094e-31  # kg
    e = 1.602e-19  # C
    hbar = 1.0546e-34  # J*s
    V = 20  # eV
    E_range = np.linspace(E_i, E_f, 1000)

    # Calculate the energies, then plot them
    odd_states = np.tan(np.sqrt((w ** 2 * m_e * e * E_range) / (2 * hbar ** 2))) + np.sqrt(E_range / (V - E_range))
    even_states = np.tan(np.sqrt((w ** 2 * m_e * e * E_range) / (2 * hbar ** 2))) - np.sqrt((V-E_range) / E_range)

    pyp.plot(E_range, odd_states, label="odd states")
    pyp.plot(E_range, even_states, label="even states")
    pyp.title("Problem #1")
    pyp.ylabel("Psi")
    pyp.xlabel("Energy (eV)")
    pyp.legend()
    pyp.grid()
    pyp.show()

    # Now go through and use bisection to calculate the first six energies,
    # alternating which function we call according to whether it is even or odd
    print(bisection_method(func_b, 0.01, 0.35, EPSILON_1))  # Energy level 1, odd
    print(bisection_method(func_a, 0.65, 2.84, EPSILON_1))  # Energy level 2, even
    print(bisection_method(func_b, 2.8, 2.9, EPSILON_1))  # etc.
    print(bisection_method(func_a, 4.9, 5.1, EPSILON_1))
    print(bisection_method(func_b, 7.8, 8.0, EPSILON_1))
    print(bisection_method(func_a, 10.9, 11.3, EPSILON_1))


def poly(x):
    return 924*x**6 - 2772*x**5 + 3150*x**4 - 1680*x**3 + 420*x**2 - 42*x + 1


def poly_prime(x):
    # Taken from Wolfram-Alpha
    return 42*(132*x**5 - 330*x**4 + 300*x**3 - 120*x**2 + 20*x -1)


def newtons_method(func, func_prime, guess, tolerance):
    delta_x = func(guess) / func_prime(guess)  # Calculate the first delta_x
    x = guess  # Set our first x to our guess
    while abs(delta_x) >= tolerance:
        # Go through and calculate iterations of x until our desired accuracy is reached
        delta_x = func(x) / func_prime(x)
        x = x - delta_x

    return x


def problem_two():
    print("Running problem #2...\n")
    x_vals = np.linspace(0, 1, 10000)  # Creating a little "x-axis"
    y_vals = poly(x_vals)  # ...and assigning y-values to it
    plt.plot(x_vals, y_vals, "6th Order Polynomial", "x", "y", "Homework_1", "Homework_1_2_1", False, False)
    # By inspecting the plot, it appears that the roots are, in order:
    root_guess = [0.037, 0.171, 0.379, 0.623, 0.833, 0.967]
    roots = []
    for guess in root_guess:
        roots.append(newtons_method(poly, poly_prime, guess, EPSILON_2))

    print("Calculated roots: " + str(roots))


def test_problem():
    # Checks to see if we get the same solution for a root of poly() using the bisection method
    # We do, so I'm happy with both implementations
    print(bisection_method(poly, 0.6, 0.65, EPSILON_2))


def run():
    print("\t\t\t\tHomework Set #1\n")
    problem_one()
    problem_two()
    # test_problem()
    print("\nDone!")
