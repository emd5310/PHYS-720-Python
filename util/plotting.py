import os
from matplotlib import pyplot as pyp
import matplotlib as mpl
mpl.use("macOSX")
# Handles making plots using matplotlib, so only one call is needed elsewhere
# MANY other plotting functions have been stripped from this one
# The version with all the fitting functions is in the Modern Lab project


def plot(x_data, y_data, title, x_label, y_label, experiment_name, filename,
         save=False, use_lims=True, x_lim=(0, 10), y_lim=(0, 10)):
    """Plots x and y data, defaults to constraining the axes from 0->10

    :param x_data: x values
    :param y_data: y values
    :param title: title for the plot
    :param x_label: x axis label for the plot
    :param y_label: y axis label for the plot
    :param experiment_name: name of the experiment performed (NO SPACES!)
    :param filename: name to save the plot as
    :param save: whether or not to save the plot
    :param use_lims: whether or not to constrain the plot
    :param x_lim: x range
    :param y_lim: y range
    :returns nothing
    """
    os.chdir("/Users/evandonohue/PycharmProjects/Modern_Lab/")
    pyp.plot(x_data, y_data)
    pyp.title(title)
    pyp.ylabel(y_label)
    pyp.xlabel(x_label)
    if use_lims:
        pyp.xlim(x_lim[0], x_lim[1])
        pyp.ylim(y_lim[0], y_lim[1])
    pyp.grid()
    print("Plotted " + title)
    if save:
        os.chdir("/Users/evandonohue/PycharmProjects/Computational Physics/")
        pyp.savefig("output/" + experiment_name + "/" + filename + ".png")
        print("Saved to: " + os.getcwd() + "/output/" + experiment_name + "/" + filename + ".png\n")
    pyp.show()
