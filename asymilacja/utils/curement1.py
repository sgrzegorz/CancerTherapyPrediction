import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from numpy.polynomial import Legendre


def curement_function(x, threatment_start, threatment_end):
    if (threatment_start >= threatment_end):
        raise RuntimeError("x1 >= x2")
    if x < threatment_start or x > threatment_end:
        return 0

    t = np.arange(threatment_start, threatment_end)
    # func = Legendre.fit([threatment_start, threatment_end/5,threatment_end / 2.,threatment_end,threatment_end+5], [1.0,  1 / 3.,1/5, 0.0,0], deg=1)
    func = Legendre.fit([threatment_start,threatment_end], [1.0,  0.0], deg=1)

    return func(x)

if __name__ == '__main__':
    threatment_start=0
    threatment_end=40
    t = np.arange(-20,threatment_end+30)
    c = [curement_function(i,threatment_start,threatment_end) for i in t]
    plt.plot(t,c)
    plt.show()
