import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d

##############################Curement 1#######################################

def curement_function(x,threatment_start,threatment_end):
    if(threatment_start >= threatment_end):
        raise RuntimeError("x1 >= x2")
    if x < threatment_start or x >= threatment_end:
        return 0

    t = np.arange(threatment_start, threatment_end+1)
    solution = odeint(model, 1, t)
    ca_func = interp1d(t, solution[:, 0], 'cubic')
    return ca_func(x)

def model(C, t=None):
    dcdt = - 1/14 * C
    return dcdt

##############################Curement 2#######################################
