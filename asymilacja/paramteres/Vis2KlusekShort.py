import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import odeint

from asymilacja.model.Cancer2KlusekShort import CancerModel


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


def plot_gt(path):
 df = pd.read_csv(path)
 figure, axis = plt.subplots(2, 1)
 p1 = axis[0]
 p2 = axis[1]

 p1.plot(df.t, df.prolif_cells, 'g', label="P")
 p1.plot(df.t, df.dead_cells, 'b', label="Q")
 p1.legend()

 p2.plot(df.t, df.curement, 'y', label="P")
 plt.show()


def plot_simple_results(pred,origin_path):

 K = pred['K']
 KDE = pred['KDE']
 eta = pred['eta']
 gamma_p = pred['gamma_p']
 gamma_q = pred['gamma_q']
 k_pq = pred['k_pq']
 lambda_p = pred['lambda_p']

 figure, axis = plt.subplots(2, 1)
 p1=axis[0]
 p2 =axis[1]

 m1 = CancerModel(lambda_p,gamma_q,gamma_p,KDE,k_pq,K,eta)

 df = pd.read_csv(origin_path)
 X0 = [df['prolif_cells'].iloc[0], df['dead_cells'].iloc[0], df['curement'].iloc[0]]

 t = np.arange(0,len(df.t))
 x = odeint(m1.model,X0, t)
 P = x[:, 0]
 Q = x[:, 1]
 C = x[:, 2]

 p1.plot(t, P, 'g', label="P")
 p1.plot(t, Q,  'b',label="Q")
 p1.legend()

 p2.plot(t, C, 'y', label="P")
 plt.show()

pred={'K': 50.255821611397344, 'KDE': 0.012609686392283898, 'eta': 0.7957916892534629, 'gamma_p': 5.360502625576902, 'gamma_q': 18.254004359130974, 'k_pq': 8.14782487806399e-06, 'lambda_p': 0.0060442423987455435}
plot_simple_results(pred,"asymilacja/trening/okres.csv")

plot_gt("asymilacja/trening/okres.csv")