# fit a straight line to the economic data
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot


# define the true objective function
from asymilacja.utils.datasets import ribba_dataset




# load the dataset
df = ribba_dataset('data/ribba/fig4.csv')



# choose the input and output variables
x, y = df.t.tolist(), df.mtd.tolist()
from numpy.polynomial import Chebyshev
c = Chebyshev.fit(x, y, deg=6)

t= np.arange(-12,60)
plt.scatter(df.t.tolist(), df.mtd.tolist())
plt.plot(t,c(t))
plt.show()
# # curve fit
# popt, _ = curve_fit(objective, x, y)
# # summarize the parameter values
# a, b = popt
# print('y = %.5f * x + %.5f' % (a, b))
# # plot input vs output
# pyplot.scatter(x, y)
# # define a sequence of inputs between the smallest and largest known inputs
# x_line = arange(min(x), max(x), 1)
# # calculate the output for the range
# y_line = objective(x_line, a, b)
# # create a line plot for the mapping function
# pyplot.plot(x_line, y_line, '--', color='red')
# pyplot.show()