import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint


def f(y, t, paras):
    """
    Your system of differential equations
    """
    [P, N, C] = y
    try:
        lambda_p = paras['lambda_p'].value
        gamma_q = paras['gamma_q'].value
        gamma_p = paras['gamma_p'].value
        KDE = paras['KDE'].value
        k_pq = paras['k_pq'].value
        K = paras['K'].value
        eta = paras['eta'].value
    except KeyError:
        lambda_p, gamma_q,gamma_p,KDE,k_pq,K,eta = paras

    dCdt = -KDE * C
    dPdt = lambda_p * P  - k_pq * P - gamma_p * C * KDE * P
    dNdt = k_pq * P - gamma_q * C * KDE * N
    return [dPdt, dNdt, dCdt]


def g(t, x0, paras):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    x = odeint(f, x0, t, args=(paras,))
    return x


def residual(ps, ts, data):
    x0 = ps['P'].value, ps['N'].value, ps['C'].value
    model = g(ts, x0, ps)
    return (model - data).ravel()


df = pd.read_csv("data/klusek/out/okres.csv")
P_true = list(df.prolif_cells)
N_true = list(df.dead_cells)
C_true = list(df.curement)
t_true = np.arange(0,len(df.t))

plt.scatter(t_true, P_true,color='red', label='P truth')
plt.scatter(t_true, N_true, color='green', label='N truth')
plt.scatter(t_true, C_true, color='yellow', label='C truth')
# df.loc[df['prolif_cells'].idxmin()]['iteration']
threatment_end = df['prolif_cells'].idxmin()

df = df[df.index < threatment_end]
df = df[df.index % 5 == 0]
P = list(df.prolif_cells)
N = list(df.dead_cells)
C = list(df.curement)
t = np.arange(0,len(df.t))

plt.scatter(t, P,color='blue', label='P taken')
plt.scatter(t, N, color='blue', label='N taken')
plt.scatter(t, C, color='blue', label='C taken')



# df = df.sample(n=40, random_state=1).sort_values(by=['iteration'])
# df = df[df.index % 10 == 0]
# df = df[df.t < 5]

P = list(df.prolif_cells)
N = list(df.dead_cells)
C = list(df.curement)


# initial conditions
y0 = [P[0], N[0], C[0]]



# measured data
t_measured = np.linspace(0, 9, 10)
x2_measured = np.array([P,N,C]).T

# plt.figure()
# plt.scatter(t_measured, x2_measured, marker='o', color='b', label='measured data', s=75)

# set parameters including bounds; you can also fix parameters (use vary=False)
params = Parameters()
params.add('P', value=y0[0], vary=False)
params.add('N', value=y0[1], vary=False)
params.add('C', value=y0[2], vary=False)
params.add('lambda_p', value=0.2, min=0.0001, max=1.)
params.add('gamma_q', value=0.3, min=0.0001, max=1.)
params.add('gamma_p', value=0.3, min=0.0001, max=1.)
params.add('KDE', value=0.3, min=0.0001, max=1.)
params.add('k_pq', value=0.3, min=0.0001, max=1.)
params.add('K', value=0.3, min=0.0001, max=1.)
params.add('eta', value=0.3, min=0.0001, max=1.)

# fit model
result = minimize(residual, params, args=(t, x2_measured), method='leastsq')  # leastsq nelder
# check results of the fit
data_fitted = g(t_true, y0, result.params)


# plot fitted data
plt.plot(t_true, data_fitted[:, 0], '-', linewidth=2, color='red', label='fitted data')
plt.plot(t_true, data_fitted[:, 1], '-', linewidth=2, color='green', label='fitted data')
plt.plot(t_true, data_fitted[:, 2], '-', linewidth=2, color='yellow', label='fitted data')

plt.legend()
# plt.xlim([0, max(t)])
# plt.ylim([0, 1.1 * max(data_fitted[:, 1])])
# display fitted statistics
report_fit(result)

plt.show()