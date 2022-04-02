import math
import pyabc

import numpy as np
from scipy.integrate import odeint
import scipy.stats as st
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy.signal import argrelextrema

from asymilacja.model.datasets import ribba_dataset

# pyabc.settings.set_figure_params('pyabc')  # for beautified plots
from asymilacja.model.CancerModelClass2 import CancerModel
from asymilacja.model.datasets import klusek_dataset


df = klusek_dataset("data/klusek/2dawki.txt")
df1 = df.copy()
maximums = argrelextrema(df.prolif_cells.to_numpy(), np.greater)[0]
t = [df.t[i] for i in maximums]
df = df[(df.t > t[0])  &  (df.t < t[1])]
# plt.plot(df.t,df.prolif_cells, label = "prolif")
# plt.plot(df.t,df.dead_cells, label = "prolif")
# plt.show()

minimum = 70
t = [df1.t[i] for i in maximums]

def curement_function(x,x1,x2):
    y1 = 1
    y2 = 0
    if(x1 == x2):
        raise RuntimeError("x1 == x2")
    if x < x1 or x > x2:
        return 0
    else:
        return (y2 - y1) / (x2-x1)* x + (x2*y1 - x1*y2)/(x2-x1)

curement = [curement_function(x,58.33333,69.16667) for x in df.t]
count = len(df["t"])


print()
P = list(df.prolif_cells)
Q = list(df.dead_cells)
C = curement
observation = np.array([P,Q,C],dtype=float)



def model(parameters):
    X0 = [parameters["P"], parameters["Q"], parameters["C"]]

    lambda_p = parameters["lambda_p"]
    gamma_q = parameters["gamma_q"]
    gamma_p = parameters["gamma_p"]
    KDE = parameters["KDE"]
    k_pq = parameters["k_pq"]
    K = parameters["K"]
    eta = parameters["eta"]


    m1 = CancerModel(lambda_p,gamma_q,gamma_p,KDE,k_pq,K,eta)
    # t = m1.time_interval(0,200)
    t = list(df.t)

    x = odeint(m1.model, X0, t)

    P = x[:, 0]
    Q = x[:, 1]
    C = x[:, 2]

    return {"data" : np.array([P,Q,C],dtype=float)}


def distance(x, y):
    dif = 0
    for i in range(count):
        dif += abs(x["data"][0][i] - y["data"][0][i])
        dif += abs(x["data"][1][i] - y["data"][1][i])
        dif += abs(x["data"][2][i] - y["data"][2][i])


    return dif


prior = pyabc.Distribution(lambda_p=pyabc.RV("uniform", 1, 20), gamma_q=pyabc.RV("uniform", 1.0e-7, 3.0e-7), gamma_p=pyabc.RV("uniform", 0.1, 1), KDE=pyabc.RV("uniform", 0.01, 0.5),
                           k_pq=pyabc.RV("uniform", 0.1, 0.7), K=pyabc.RV("uniform", 0.01, 0.5),eta=pyabc.RV("uniform", 0.01, 0.5),
                           P= pyabc.RV("uniform", 1, 1e7),Q= pyabc.RV("uniform", 1, 1e7),C= pyabc.RV("uniform", 0.0, 1.1))

abc = pyabc.ABCSMC(model, prior, distance, population_size=4)

db_path = ("sqlite:///" +  os.path.join(tempfile.gettempdir(), "test.db"))

abc.new(db_path, {"data": observation})

history = abc.run(minimum_epsilon=100, max_nr_populations=40)

history is abc.history

posterior2 = pyabc.MultivariateNormalTransition()
posterior2.fit(*history.get_distribution(m=0))

t_params = posterior2.rvs()

t_params = np.array([[param, t_params[param]] for param in t_params])

print(t_params)