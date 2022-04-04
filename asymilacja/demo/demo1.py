from asymilacja.model.CancerModelClass1 import CancerModel
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from asymilacja.model.datasets import ribba_dataset

patient = ribba_dataset('/data/fig4.csv')
patient = patient[patient["t"]>0]


lambda_p = 0.6326
delta_qp = 0.6455
gamma_q = 1.3495
gamma_p = 4.6922
KDE = 0.10045
k_qpp = 0.0
k_pq = 0.43562
K = 192.418

m1 = CancerModel(lambda_p, delta_qp, gamma_q, gamma_p, KDE, k_qpp, k_pq, K)

P = 4.7279
Q = 48.5147
Q_p = 0.0
C = 0.0

x0 = [P, Q, Q_p, C]
t = m1.time_interval(0, 120) # start, end, steps
x = odeint(m1.model,x0, t)

P = x[:, 0]
Q = x[:, 1]
Q_p = x[:, 2]
C = x[:, 3]
M = x[:, 4]

def cancer_plot(t,P,Q,Q_p):
    plt.plot(t,P,label="P",color='g')
    plt.plot(t,Q,label="Q",color='r')
    plt.plot(t,Q_p,label="Q_p",color='black')
    plt.plot(t,P+Q+Q_p, label="P+Q+Q_p",color='w')
cancer_plot(t,P,Q,Q_p)


plt.title('Cancer')
plt.legend(loc="lower right")
plt.xlabel("months")
plt.ylabel('volume [mm]')
cancer_plot(t,P,Q,Q_p)




plt.title('Curement')
plt.plot(t,C,color='b')
plt.plot(t,C,color='b')
plt.xlabel("months")
plt.show()

