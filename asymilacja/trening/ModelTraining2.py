import math
import pyabc

import numpy as np
from scipy.integrate import odeint
import scipy.stats as st
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


# pyabc.settings.set_figure_params('pyabc')  # for beautified plots
from sklearn import preprocessing

from asymilacja.model.CancerModelClass2 import CancerModel
from asymilacja.model.datasets import klusek_dataset
from asymilacja.paramteres.Parameters2 import plot_simple_results

df = pd.read_csv("asymilacja/trening/okres.csv")

count = len(df["t"])



P = list(df.prolif_cells)
Q = list(df.dead_cells)
C = list(df.curement)
observation = np.array([P,Q,C],dtype=float)

X0 = [P[0], Q[0], C[0]]



def model(parameters):

    lambda_p = parameters["lambda_p"]
    gamma_q = parameters["gamma_q"]
    gamma_p = parameters["gamma_p"]
    KDE = parameters["KDE"]
    k_pq = parameters["k_pq"]
    K = parameters["K"]
    eta = parameters["eta"]


    m1 = CancerModel(lambda_p,gamma_q,gamma_p,KDE,k_pq,K,eta)
    # t = m1.time_interval(0,200)
    t = np.arange(0,len(df.t))

    x = odeint(m1.model, X0, t)

    P = x[:, 0]
    Q = x[:, 1]
    C = x[:, 2]

    return {"data" : np.array([P,Q,C],dtype=float)}


def distance(x, y):
    return np.absolute(x["data"][0]- y["data"][0]).sum()


def distance1(x, y):
    return np.absolute(x["data"][1]- y["data"][1]).sum()


def distance2(x, y):
    return np.absolute(x["data"][2]- y["data"][2]).sum()


dist = pyabc.distance.AggregatedDistance([distance,distance1,distance2])

# prior = pyabc.Distribution(lambda_p=pyabc.RV("uniform", 0.0, 1.0), gamma_q=pyabc.RV("uniform", 0.0, 1.0), gamma_p=pyabc.RV("uniform", 0.0, 1.0), KDE=pyabc.RV("uniform", 0.0, 1.0),
#                            k_pq=pyabc.RV("uniform", 0.0, 1.0), K=pyabc.RV("uniform", 0.0, 1.0)
#                            )

prior = pyabc.Distribution(#a=pyabc.RV("uniform", 1, 4),
                           KDE=pyabc.RV("uniform", 0, 20), K=pyabc.RV("uniform", 50, 250), k_pq=pyabc.RV("uniform", 0, 20),
                            lambda_p=pyabc.RV("uniform", 0, 20), gamma_p=pyabc.RV("uniform", 0, 20),
                           gamma_q=pyabc.RV("uniform", 0, 20),eta=pyabc.RV("uniform",0,1))

abc = pyabc.ABCSMC(model, prior, dist, population_size=100)

db_path = ("sqlite:///" +  os.path.join(tempfile.gettempdir(), "test.db"))

abc.new(db_path, {"data": observation})

history = abc.run(minimum_epsilon=10, max_nr_populations=350)

history is abc.history
run_id = history.id

posterior2 = pyabc.MultivariateNormalTransition()
posterior2.fit(*history.get_distribution(m=0))

t_params = posterior2.rvs()
#
# posterior2 = pyabc.MultivariateNormalTransition()
# posterior2.fit(*history.get_distribution(m=0))
#
# t_params = posterior2.rvs()
#
# t_params = np.array([[param, t_params[param]] for param in t_params])
#
# print(t_params)
#
# for i in range(50):
#     abc_continued = pyabc.ABCSMC(model, prior, distance)
#     abc_continued.load(db_path, run_id)
#     # try:
#     history = abc_continued.run(minimum_epsilon=.1, max_nr_populations=4)
#
#     posterior2 = pyabc.MultivariateNormalTransition()
#     posterior2.fit(*history.get_distribution(m=0))
#
#     t_params = posterior2.rvs()
#
#     print(t_params)
#
#     t_params = np.array([[param, t_params[param]] for param in t_params])
#
#     print(t_params)
    # except AssertionError as e:
    #     print(e)
    #     break


print(t_params)
# import pickle
# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(t_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     df1 = pickle.load(handle)

t_params1 = {}
for key, val in t_params:
    t_params1[key] = float(val)
# plot_results(t_params1,"data/klusek/2dawki.txt")
plot_simple_results(t_params1,"data/klusek/2dawki.txt")




# t_params = np.array([[param, t_params[param]] for param in t_params])




