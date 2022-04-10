import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from asymilacja.utlis.datasets import ribba_dataset
from asymilacja.utlis.curement import curement_function
from asymilacja.model.Cancer1Ribba import CancerModel
from numpy.polynomial import Chebyshev



def full_plot(P0, Q0, Q_p0, C0,steps_backward,steps_forward,pred):

    X0 = [P0, Q0, Q_p0, C0]

    m1 = CancerModel(lambda_p=pred['lambda_p'], delta_qp=pred["delta_qp"], gamma_q=pred['gamma_q'],
                     gamma_p=pred['gamma_p'], KDE=pred["KDE"], k_qpp=pred['k_qpp'], k_pq=pred['k_pq'], K=pred["K"])

    t1 = np.linspace(0, -steps_backward, steps_backward + 1)
    x = odeint(m1.model, X0, t1)
    P1 = x[:, 0]
    Q1 = x[:, 1]
    Q_p1 = x[:, 2]
    C1 = x[:, 3]

    t2 = np.linspace(0, steps_forward, steps_forward + 1)
    X0 = [P1[0], Q1[0], Q_p1[0], 1.0]
    x = odeint(m1.model, X0, t2)
    P2 = x[:, 0]
    Q2 = x[:, 1]
    Q_p2 = x[:, 2]
    C2 = x[:, 3]

    t = np.concatenate((np.flip(t1), t2))
    P = np.concatenate((np.flip(P1), P2))
    Q = np.concatenate((np.flip(Q1), Q2))
    Q_p = np.concatenate((np.flip(Q_p1), Q_p2))
    C = np.concatenate((np.flip(C1), C2))

    plt.plot(t, P + Q + Q_p)
    plt.show()


# df = ribba_dataset('data/ribba/fig4.csv')
# # df = df[df["t"]>0]
# threatment_start = 0
# threatment_end = 26
# df['curement'] = [curement_function(i,threatment_start,threatment_end) for i in df.t.tolist()]
#
# start = df.t.tolist()[0]
# end = df.t.tolist()[-1]
# t = np.arange(start,end)
# # plt.plot(df)
# # plt.show()
# fit = Chebyshev.fit(df.t.tolist(), df.mtd.tolist(), deg=6)
# mtd = [fit(i) for i in t]
# curement = [curement_function(i,threatment_start,threatment_end) for i in t]

pred = {'K': 196.91524201781607, 'KDE': 0.02993977035657992, 'delta_qp': 0.518550701246765,
        'gamma_p': 14.179606546920718, 'gamma_q': 2.85767841085851, 'k_pq': 0.6451121356194429,
        'k_qpp': 0.010125204708275796, 'lambda_p': 1.0082846643230274}

mtd = [44.531545866621514]
P0 = 0.1 *mtd[0]
Q0 = mtd[0] - P0
Q_p0 = 0.0
C0 = 0.0
full_plot(P0,Q0,Q_p0,C0,20,200,pred)
