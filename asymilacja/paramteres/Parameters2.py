import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelextrema

from asymilacja.model.CancerModelClass2 import CancerModel
from asymilacja.model.datasets import klusek_dataset

# def plot_results(pred,origin_path):
#
#
#  # t = m1.time_interval(-20, 0) # start, end, steps
#
#  df = klusek_dataset(origin_path)
#  p1.plot(df.t, list(df.prolif_cells), 'g:',label="P")
#  p1.plot(df.t, list(df.dead_cells), 'b:',label="Q")
#  # p2.plot(df.t, list(df.curement), 'y:')
#  p1.set(xlim=(0, list(df.t)[-1]))
#  p2.set(xlim=(0, list(df.t)[-1]))
#
#
#  maximums = argrelextrema(df.prolif_cells.to_numpy(), np.greater)[0]
#  maxs = [df.t[i] for i in maximums]
#  df = df[(df.t > maxs[0])  &  (df.t < maxs[1])]
#  t = df.t




def plot_simple_results(pred,origin_path):
 C = 1.0
 K = pred['K']
 KDE = pred['KDE']
 P0 = pred['P']
 Q0= pred['Q']
 eta = pred['eta']
 gamma_p = pred['gamma_p']
 gamma_q = pred['gamma_q']
 k_pq = pred['k_pq']
 lambda_p = pred['lambda_p']

 figure, axis = plt.subplots(2, 1)
 p1=axis[0]
 p2 =axis[1]

 m1 = CancerModel(lambda_p,gamma_q,gamma_p,KDE,k_pq,K,eta)
 x0 = [P0, Q0, C]

 df = klusek_dataset(origin_path)
 t = df.t
 x = odeint(m1.model,x0, t)
 P = x[:, 0]
 Q = x[:, 1]
 C = x[:, 2]

 p1.plot(t, P, 'g', label="P")
 p1.plot(t, Q,  'b',label="Q")
 p1.legend()

 p2.plot(t, C, 'y', label="P")
 plt.show()

pred = {'K': 109.27498625317781, 'KDE': 0.6011433707451191, 'P': 49.33540506540792, 'Q': 4.189374073842397, 'eta': 0.23604759815002457, 'gamma_p': 9.263211986317149, 'gamma_q': 15.073014403126052, 'k_pq': 13.865416769630965, 'lambda_p': 5.61378685096156}

# plot_simple_results(pred,"data/klusek/2dawki.txt")

