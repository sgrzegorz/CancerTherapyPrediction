import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

from asymilacja.model.Cancer2KlusekShort import CancerModel
from asymilacja.utils.datasets import klusek_dataset
from scipy.signal import argrelextrema


df = klusek_dataset("data/klusek/patient4/2dawki.txt")

# plt.plot(df.t,df.prolif_cells, label = "prolif")
# plt.plot(df.t,df.dead_cells, label = "prolif")
# maximums = argrelextrema(df.prolif_cells.to_numpy(), np.greater)
# x = [df.t[i] for i in maximums]
# y = [df.prolif_cells[i] for i in maximums]
# plt.scatter(x,y,color='red')
# plt.show()

maximums = argrelextrema(df.prolif_cells.to_numpy(), np.greater)[0]
maximums1 = [df.t[i] for i in maximums]
df = df[(df.t > maximums1[0])  &  (df.t < maximums1[1])]

print(df)

lambda_p = 0.6326
delta_qp = 0.6455
gamma_q = 1.3495
gamma_p = 4.6922
KDE = 0.10045
k_qpp = 0.0
k_pq = 0.43562
K = 192.418
eta = 0.3

m1 = CancerModel(lambda_p,gamma_q,gamma_p,KDE,k_pq,K,eta)
P = 124.7279
Q = 48.5147
Q_p = 0.0
C = 0.8
t = df.t
x0 = [P, Q, C]
# t = m1.time_interval(-20, 0) # start, end, steps
x = odeint(m1.model,x0, t)

P = x[:, 0]
Q = x[:, 1]
C = x[:, 2]

plt.plot(t, P, label="P", color='g')
plt.plot(t, Q, label="Q", color='black')
plt.legend()

plt.show()

plt.plot(t, C, label="C", color='blue')
plt.legend()
plt.show()


