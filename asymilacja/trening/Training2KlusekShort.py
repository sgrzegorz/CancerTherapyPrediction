import pyabc

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import tempfile
import os
import pandas as pd

# pyabc.settings.set_figure_params('pyabc')  # for beautified plots

from asymilacja.model.Cancer2KlusekShort import CancerModel
from asymilacja.paramteres.Vis2KlusekShort import plot_simple_results

df = pd.read_csv("data/klusek/patient4/okres.csv")

count = len(df["t"])



P = list(df.prolif_cells)
N = list(df.dead_cells)
C = list(df.curement)
observation = np.array([P,N,C],dtype=float)

t = np.arange(0,len(df.t))
X0 = [P[0], N[0], C[0]]


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

    x = odeint(m1.model, X0, t)

    P = x[:, 0]
    N = x[:, 1]
    C = x[:, 2]

    return {"data" : np.array([P,N,C],dtype=float)}


def distance(x, y):
    return np.absolute(x["data"][0]- y["data"][0]).sum()


def distance1(x, y):
    return np.absolute(x["data"][1]- y["data"][1]).sum()


def distance2(x, y):
    return np.absolute(x["data"][2]- y["data"][2]).sum()


dist = pyabc.distance.AggregatedDistance([distance,distance1,distance2])

prior = pyabc.Distribution(
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

print(t_params)


data = model(t_params)["data"]
P = data[0,:]
N = data[1,:]
C = data[2,:]
plt.title("out")
plt.plot(t,P,label='P')
plt.plot(t,N,label='N')
plt.legend()
plt.show()

# {'K': 222.26912030083898, 'KDE': 0.0022379173337168695, 'eta': 0.9398272992772523, 'gamma_p': 10.704809708433405, 'gamma_q': 19.758732923512987, 'k_pq': 1.6247407229958707e-05, 'lambda_p': 4.4135534550830785e-11}

# t_params = np.array([[param, t_params[param]] for param in t_params])




