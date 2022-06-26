import pyabc

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import tempfile
import os
import pandas as pd

# pyabc.settings.set_figure_params('pyabc')  # for beautified plots

from asymilacja.model.Cancer2KlusekShort1 import CancerModel
from asymilacja.paramteres.Vis2KlusekShort import plot_simple_results

df = pd.read_csv("data/klusek/patient4/2dawki_p1.csv")

count = len(df["t"])
steps_backward = count-1


P = list(df.prolif_cells)
Q = list(df.dead_cells)
C = list(df.curement)
observation = np.array([P,Q,C],dtype=float)

# X0 = [P[-1], Q[-1], C[-1]]
X0 = [P[0], Q[0], C[0]]
t = np.arange(0, count)


def model(parameters):

    lambda_p = parameters["lambda_p"]
    gamma_q = parameters["gamma_q"]
    gamma_p = parameters["gamma_p"]
    KDE = parameters["KDE"]
    k_pq = parameters["k_pq"]
    K = 1
    eta = parameters["eta"]


    m1 = CancerModel(lambda_p,gamma_q,gamma_p,KDE,k_pq,K,eta)

    # t = np.linspace(0, -steps_backward, steps_backward+1)
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


prior = pyabc.Distribution(
                           KDE=pyabc.RV("uniform", 0.1, 1e6), k_pq=pyabc.RV("uniform", 0.1, 1e6),
                            lambda_p=pyabc.RV("uniform", 0.1, 2e6), gamma_p=pyabc.RV("uniform", 0.1, 1e6),
                           gamma_q=pyabc.RV("uniform", 0.1, 1e6),eta=pyabc.RV("uniform",0.1,1e6))

abc = pyabc.ABCSMC(model, prior, dist, population_size=100)

db_path = ("sqlite:///" +  os.path.join(tempfile.gettempdir(), "test.db"))

abc.new(db_path, {"data": observation})

history = abc.run(minimum_epsilon=10, max_nr_populations=100)

history is abc.history
run_id = history.id

posterior2 = pyabc.MultivariateNormalTransition()
posterior2.fit(*history.get_distribution(m=0))

t_params = posterior2.rvs()

print(t_params)


data = model(t_params)["data"]
P = data[0,:]
Q = data[1,:]
C = data[2,:]

# t = data[0,:]

plt.title("out")
plt.plot(t,P,label='P')
plt.plot(t,Q,label='Q')
plt.legend()
plt.show()

plt.plot(t,C)
plt.show()




