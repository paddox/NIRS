import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction

def lineplot(x_data, y_data, x_label="", y_label="", title=""):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot(x_data, y_data, lw = 2, color = '#539caf', alpha = 1)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    plt.show()

def scatterplot(x_data, y_data, x_label="", y_label="", title="", color = "r", yscale_log=False):

    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, s = 10, color = color, alpha = 0.75)

    if yscale_log == True:
        ax.set_yscale('log')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.show()

def f(x,y):
    return Fraction(x**2 - 5/3*x*y+3/5*x-4*y+2/3)
    #return Fraction(1/2*x-y)

def enum():
    i = 1
    j = 1
    array = []
    while i < 100 and j < 15:
        if i % 2 == 1 and j == 1:
            array.append(Fraction(i,j))
            i += 1
            continue
        if i == 1 and j % 2 == 0:
            array.append(Fraction(i,j))
            j += 1
            continue
        if (i + j) % 2 == 1:
            array.append(Fraction(i,j))
            i -= 1
            j += 1
            continue
        if (i + j) % 2 == 0:
            array.append(Fraction(i,j))
            i += 1
            j -= 1
            continue
        break
    return array