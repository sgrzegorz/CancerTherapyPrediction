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




print()
# P = df.prolif_cells
# Q = df.dead_cells
# C = 0.0
# observation = {"t" : df["t"].tolist(), "mtd" : df["mtd"].tolist()}
#
# count = len(observation["t"])
#
#
# def model(parameters):
#     X0 = [parameters["P"], parameters["Q"], parameters["Q_p"], parameters["C"]]
#
#     lambda_p = parameters["lambda_p"]
#     delta_qp = parameters["delta_qp"]
#     gamma_q = parameters["gamma_q"]
#     gamma_p = parameters["gamma_p"]
#     KDE = parameters["KDE"]
#     k_qpp = parameters["k_qpp"]
#     k_pq = parameters["k_pq"]
#     K = parameters["K"]
#
#
#     m1 = CancerModel(lambda_p, delta_qp, gamma_q, gamma_p, KDE, k_qpp, k_pq, K)
#     t = m1.time_interval(0,200)
#
#     x = odeint(m1.model, X0, t)
#
#     P = x[:, 0]
#     Q = x[:, 1]
#     Q_p = x[:, 2]
#     C = x[:, 3]
#
#     return {"P" : P, "Q" : Q, "Q_p" : Q_p,"C": C}
#
#
# def distance(x, y):
#     dif = 0
#     for i in range(count):
#         dif += abs(x["data"][i] - y["data"][i])
#     return dif
#
#
# prior = pyabc.Distribution(lambda_p=pyabc.RV("uniform", 1, 20), delta_qp=pyabc.RV("uniform", 1.00e-8, 3.00e-8), gamma_q=pyabc.RV("uniform", 1.0e-7, 3.0e-7), gamma_p=pyabc.RV("uniform", 0.1, 1), KDE=pyabc.RV("uniform", 0.01, 0.5), k_qpp=pyabc.RV("uniform", 0.01, 0.5),
#                            k_pq=pyabc.RV("uniform", 0.1, 0.7), K=pyabc.RV("uniform", 0.01, 0.5))
#
# abc = pyabc.ABCSMC(model, prior, distance, population_size=4)
#
# db_path = ("sqlite:///" +  os.path.join(tempfile.gettempdir(), "test.db"))
#
# abc.new(db_path, {"data": observation})
#
# history = abc.run(minimum_epsilon=100, max_nr_populations=4)
#
# history is abc.history
