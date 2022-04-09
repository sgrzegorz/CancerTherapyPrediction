import pyabc

from scipy.integrate import odeint
import tempfile
import os
# %matplotlib inline

# pyabc.settings.set_figure_params('pyabc')  # for beautified plots
from asymilacja.model.Cancer1Ribba import CancerModel

import numpy as np
import pandas as pd


patient = pd.read_csv('data/ribba/sztucznyDemo1.csv')

observation = np.array([ patient["P"].tolist(),  patient["Q"].tolist(), patient["Q_p"].tolist(), patient["C"].tolist()],dtype=float)
count = len(patient["P"].tolist())

X0 = [patient["P"][0], patient["Q"][0], patient["Q_p"][0], patient["C"][0]]


def model(parameters):

    lambda_p = parameters["lambda_p"]
    delta_qp = parameters["delta_qp"]
    gamma_q = parameters["gamma_q"]
    gamma_p = parameters["gamma_p"]
    KDE = parameters["KDE"]
    k_qpp = parameters["k_qpp"]
    k_pq = parameters["k_pq"]
    K = parameters["K"]


    m1 = CancerModel(lambda_p,delta_qp,gamma_q,gamma_p,KDE,k_qpp,k_pq,K)
    t = np.arange(len(patient["P"]))

    x = odeint(m1.model, X0, t)

    P = x[:, 0]
    Q = x[:, 1]
    Q_p = x[:, 2]
    C = x[:, 3]

    return {"data" :np.array([ P,  Q,  Q_p,C],dtype=float)}


def distance(x, y):
    dif = 0
    for i in range(count):
        dif += abs(x["data"][0][i] - y["data"][0][i])
    return dif


def distance1(x, y):
    dif = 0
    for i in range(count):
        dif += abs(x["data"][1][i] - y["data"][1][i])
    return dif

def distance2(x, y):
    dif = 0
    for i in range(count):
        dif += abs(x["data"][2][i] - y["data"][2][i])
    return dif

def distance3(x, y):
    dif = 0
    for i in range(count):
        dif += abs(x["data"][3][i] - y["data"][3][i])
    return dif


prior = pyabc.Distribution(lambda_p=pyabc.RV("uniform", 0.01, 1), delta_qp=pyabc.RV("uniform", 0.01, 1), gamma_q=pyabc.RV("uniform", 0.01, 2), gamma_p=pyabc.RV("uniform", 0.01, 10), KDE=pyabc.RV("uniform", 0.01, 0.5), k_qpp=pyabc.RV("uniform", 0.01, 0.5), k_pq=pyabc.RV("uniform", 0.01, 0.7), K=pyabc.RV("uniform", 0.01, 200))

dist = pyabc.distance.AggregatedDistance([distance,distance1,distance2,distance3])
abc = pyabc.ABCSMC(model, prior, dist, population_size=100)

db_path = ("sqlite:///" +  os.path.join(tempfile.gettempdir(), "test.db"))

abc.new(db_path, {"data": observation})

history = abc.run(minimum_epsilon=100, max_nr_populations=300)

history is abc.history

posterior2 = pyabc.MultivariateNormalTransition()
posterior2.fit(*history.get_distribution(m=0))

t_params = posterior2.rvs()

print(t_params)