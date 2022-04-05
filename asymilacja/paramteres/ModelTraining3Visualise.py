import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import odeint

from asymilacja.model.CancerModelClass import CancerModel

parameters= {'K': 188.94642758955172, 'KDE': 0.09437746180369837, 'delta_qp': 0.7897257286288335, 'gamma_p': 9.457518420779783, 'gamma_q': 1.3771861428800967, 'k_pq': 0.4129043294229215, 'k_qpp': 0.013031352255034285, 'lambda_p': 0.605068075155125}
patient = pd.read_csv('data/ribba/sztucznyDemo1.csv')

observation = np.array([ patient["P"].tolist(),  patient["Q"].tolist(), patient["Q_p"].tolist(), patient["C"].tolist()],dtype=float)
count = len(patient["P"].tolist())

X0 = [patient["P"][0], patient["Q"][0], patient["Q_p"][0], patient["C"][0]]


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

plt.plot(t,P)
plt.plot(t,Q)
plt.plot(t,Q_p)
plt.plot(t,C)
plt.show()
print(parameters)
