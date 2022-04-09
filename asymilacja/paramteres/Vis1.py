from matplotlib import pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from asymilacja.model.Cancer2KlusekShort import CancerModel


def plot_results(pred):

 C = 1
 K = pred['K']
 KDE = pred['KDE']
 P0 = pred['P']
 Q0= pred['Q']
 # eta = pred['eta']
 gamma_p = pred['gamma_p']
 gamma_q = pred['gamma_q']
 k_pq = pred['k_pq']
 lambda_p = pred['lambda_p']
 eta = pred['eta']

 figure, axis = plt.subplots(2, 1)
 p1=axis[0]
 p2 =axis[1]

 m1 = CancerModel(lambda_p,gamma_q,gamma_p,KDE,k_pq,K,eta)
 x0 = [P0, Q0, C]

 # p1.set(xlim=(0, list(df.t)[-1]))
 # p2.set(xlim=(0, list(df.t)[-1]))

 df = pd.read_csv("data/klusek/calosc.csv")

 t = list(df.t)

 x = odeint(m1.model,x0, t)
 P = x[:, 0]
 Q = x[:, 1]
 C = x[:, 2]

 p1.plot(t, P, 'g', label="P")
 p1.plot(t, Q,  'b',label="Q")
 p1.legend()

 p2.plot(t, C, 'y', label="P")
 plt.show()

pred ={'K': 218.4971387600425, 'KDE': 16.461605448866344, 'P': 49.28355660120901, 'Q': 4.213418643110418, 'eta': 0.18382701584756397, 'gamma_p': 15.4939344114866, 'gamma_q': 9.940199517500922, 'k_pq': 5.516509459040784, 'lambda_p': 1.5812932293441018}

plot_results(pred)