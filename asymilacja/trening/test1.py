import numpy as np
from matplotlib import pyplot as plt


def unit_step_fun(x,threshold):
    return x*(1 / 2 + 1 / 2 *np.tanh(100 * (x-threshold)))



print(unit_step_fun(0.2,0.3))