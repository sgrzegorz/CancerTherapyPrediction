from asymilacja.model.Cancer1Ribba import CancerModel
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

from asymilacja.utils.datasets import ribba_dataset

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
t = m1.time_interval(-20, 0) # start, end, steps
x = odeint(m1.model,x0, t)

P = x[:, 0]
Q = x[:, 1]
Q_p = x[:, 2]
C = x[:, 3]
C_copy = C
t_copy = t


def cancer_plot(t,P,Q,Q_p):
    plt.plot(t,P,label="Proliferative",color='g')
    plt.plot(t,Q,label="Undamaged Quiescent",color='r')
    plt.plot(t,Q_p,label="Damaged Quiescent",color='black')
    plt.plot(t,P+Q+Q_p, label="P+Q+Q_p",color='w')
cancer_plot(t,P,Q,Q_p)

last = len(P)-1
x1 = [P[last], Q[last], Q_p[last], 1.0]

print('Parameters from starting treatment')
print(x1)
t = m1.time_interval(0, 200)
m1 = CancerModel(lambda_p, delta_qp, gamma_q, gamma_p, KDE, k_qpp, k_pq, K)
x = odeint(m1.model,x1, t)

P = x[:, 0]
Q = x[:, 1]
Q_p = x[:, 2]
C = x[:, 3]

plt.title('Model Ribby przed asymilacją paramtrów')
plt.legend(loc="lower right")
plt.xlabel("months")
plt.ylabel('volume [mm]')
cancer_plot(t,P,Q,Q_p)

patient = ribba_dataset('data/ribba/fig4.csv')
plt.scatter(patient.t,patient.mtd)
plt.xlabel("time [months]")
plt.ylabel("mtd [mm]")

plt.show()



plt.title('Curement')
plt.plot(t_copy,C_copy,color='b')
plt.plot(t,C,color='b')
plt.xlabel("months")
plt.show()

df = pd.DataFrame()
df['P'] = P
df['Q'] = Q
df['Q_p'] = Q_p
df['C'] = C
df.to_csv("data/ribba/sztucznyDemo.csv")