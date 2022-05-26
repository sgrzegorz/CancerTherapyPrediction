import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint

from asymilacja.utils.preprocessAndNormalize import NormalizeData


def unit_step_fun(x,threshold):
    return x*(1 / 2 + 1 / 2 *np.tanh(100 * (x-threshold)))


def f(y, t, paras):
    [P, C] = y
    lambda_p = paras['lambda_p'].value
    gamma_p = paras['gamma_p'].value
    KDE = paras['KDE'].value
    K = paras['K'].value
    eta = paras['eta'].value*paras['C0'].value
    T_death = paras['T_death'].value

    dCdt =-KDE * C
    dPdt = lambda_p * P*(1-P/K) - 1/T_death*P - gamma_p * unit_step_fun(C,eta)  * P
    return [dPdt, dCdt]


def g(t, x0, paras):
    x, info  = odeint(f, x0, t, args=(paras,),full_output=True)
    return x


def residual(ps, ts, data):
    x0 = ps['P0'].value, ps['C0'].value
    model = g(ts, x0, ps)
    return (model[:,0] - data).ravel()


df_true = pd.read_csv("data/klusek/patient202205170854/stats0.csv")
from data.klusek.patient202205141015.config import threatment_start, threatment_end,threatment2_start
threatment_time = threatment_end - threatment_start
df_true = df_true[(df_true['iteration'] >= threatment_start) & (df_true['iteration']<=threatment2_start)]

P_true = list(df_true.prolif_cells)
N_true = list(df_true.dead_cells)
t_true = list(df_true.iteration)

fig, (plt1,plt2) = plt.subplots(2,1)
fig.set_figheight(10)
plt1.plot(t_true, P_true,color='black', linewidth=1, label='model Adriana')

df = df_true
# df = df_true[df_true.index < (6/6*threatment_time)+threatment_start]
if df.empty:
    raise ValueError("No data provided!")

# warunek = []
# for i in list(df_true.index):
#     if i < threatment_end:
#         warunek.append(i % 3==0)
#     else:
#         warunek.append(i % 40==0)
# df = df_true[warunek]


# df = df[df.index % 4 == 0]

# noise = np.random.normal(0, .1, df.shape[0]) *0.05* np.max(df.prolif_cells)
# P = list(df.prolif_cells+noise)
P = list(df.prolif_cells)


N = list(df.dead_cells)
t = list(df.iteration)

plt1.scatter(t, P,color='yellow', label='przedział uczenia')
plt1.set_title("Rys1 Komórki proliferatywne")
# plt.scatter(t, N, color='blue', label='N taken')
plt2.scatter(t, np.repeat(0,len(t)), color='yellow', label='przedział uczenia')
plt2.set_title("Rys2 Lekarstwo")


params = Parameters()
params.add('P0', value=P[0], vary=False)
params.add('C0', min=3, max=10)
params.add('gamma_p', min=0.0005, max=0.002)
params.add('K', value=0.35e6, min=0.3e6, max=0.5e6)
params.add('T_death', value=1.5e7, min=1e8, max=1.e9)
params.add('eta', value=0.1, min=0.05, max=0.2) #uwaga eta jest modyfikowana w f, min=0.1 bedzie min=0.1*C0
params.add('KDE',  expr=f'-ln(eta)/({threatment_time}+200)')
params.add('lambda_p', min=0.0005, max=0.002)

t_measured = np.linspace(threatment_start,threatment_end,num=4)
t_measured = np.concatenate((t_measured,np.linspace(threatment_end,threatment2_start,num=6)))
t_measured = np.around(t_measured)
P_measured = df.loc[t_measured,'prolif_cells'].to_list()

# fit model
result = minimize(residual, params, args=(t_measured, P_measured), method='least_squares')  # leastsq nelder
data_fitted = g(t_true, [P[0], result.params['C0'].value], result.params)

# plot fitted data
params_eta =result.params['eta'].value*result.params['C0'].value
plt1.plot(t_true, data_fitted[:, 0], '-', linewidth=1, color='green', label='model uproszczony ')
plt1.set_ylabel("cells count")
plt1.set_xlabel("time")
plt2.plot(t_true, data_fitted[:, 1], '-', linewidth=1, color='green', label='model uproszczony')
plt2.plot(t_true, np.repeat(params_eta,len(t_true)), '--', linewidth=1, color='blue', label='threshold')
plt2.set_xlabel("time\n1)poniżej poziomu threshold w modelu uproszczonym lekarstwo nie działa")

plt2.plot(t_true,[unit_step_fun(x,params_eta) for x in data_fitted[:, 1]],'--',linewidth=1,color='brown',label="efektywność lekarstwa")
plt1.legend()
plt2.legend()

# display fitted statistics
report_fit(result)


plt.show()

print(result.params.valuesdict())

#{'P0': 1578183, 'C0': 3.0000000000876295, 'gamma_p': 0.000921608661369091, 'KDE': 0.00044991509877650835, 'K': 1833437.6769648865, 'eta': 0.29999999999999993, 'psi_p': 0.0015093473209401896, 'lambda_p_m': 0.0005000000000000001, 'lambda_p': 0.0020093473209401897}