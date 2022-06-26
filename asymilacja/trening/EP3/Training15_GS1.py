import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from asymilacja.trening.utils import *


def f(y, t, paras):
    [P, C] = y
    lambda_p = paras['lambda_p'].value
    gamma_p = paras['gamma_p'].value
    KDE = paras['KDE'].value
    K = paras['K'].value
    eta = paras['eta'].value
    eta = eta*paras['C0'].value

    dCdt =-KDE * C
    dPdt = lambda_p * P*(1-P/K)  - gamma_p * unit_step_fun(C,eta)  * P
    return [dPdt, dCdt]


def g(t, x0, paras):
    x = odeint(f, x0, t, args=(paras,))
    return x

def residual(ps, ts, data):
    x0 = ps['P0'].value, ps['C0'].value
    model = g(ts, x0, ps)
    return (model[:,0] - data).ravel()


df_true = pd.read_csv("data/klusek/EP3/stats0.csv")
from data.klusek.EP3.config import threatment_start, threatment_end,threatment2_start
threatment_time = threatment_end - threatment_start
df_true = df_true[(df_true['iteration'] >= threatment_start) & (df_true['iteration']<=threatment2_start)]

P_true = list(df_true.prolif_cells)
N_true = list(df_true.dead_cells)
t_true = list(df_true.iteration)


df = df_true
# df = df_true[df_true.index < (6/6*threatment_time)+threatment_start]

if df.empty:
    raise ValueError("No data provided!")



P = list(df.prolif_cells)
t = list(df.iteration)

maxi = np.max(df.prolif_cells)

maxi = np.max(df.prolif_cells)
params = Parameters()
params.add('P0', value=P[0], vary=False)
params.add('C0', min=3, max=10)
params.add('gamma_p',value=0.003, min=0.0001, max=0.01)
params.add('K', value=1.1 * maxi, min=1.0 * maxi, max=1.7 * maxi)
# params.add('eta', expr='0.2*C0')
params.add('eta', value=0.2, min=0.1, max=0.3) #uwaga eta jest modyfikowana w f, min=0.1 bedzie min=0.1*C0
params.add('KDE', value=0.007, expr=f'-ln(eta)/({threatment_time}+200)')
params.add('lambda_p', value=0.005,min=0.0001,max=0.01)
# assimilated_parameters =
# for parName, parVal in assimilated_parameters.items():
#     params[parName].set(value=parVal)


df_1 = df.loc[df['iteration'] <= threatment_end]
df_2 = df.loc[df['iteration'] > threatment_end]
df_1 = df_1.iloc[::700,:]
df_2 = df_2.iloc[::5000,:]
df_sampled = pd.concat([df_1,df_2])
t_measured = list(df_sampled.iteration)
P_measured = list(df_sampled.prolif_cells)

result = minimize(residual, params, args=(t_measured, P_measured), method='powell')  # leastsq nelder


x0 = [P[0], result.params['C0'].value]
data_fitted = g(t_true, x0, result.params)
P_fitted = data_fitted[:, 0]
C_fitted =data_fitted[:, 1]
params_eta =result.params['eta'].value*result.params['C0'].value
plot_assimilation(t_true,P_true, P_fitted, C_fitted,params_eta,t=t,P=P,t_measured=t_measured, P_measured=P_measured)

# display fitted statistics
report_fit(result)
dopasowanie = np.linalg.norm(P_fitted - P_true, ord=2)
print(f'Least square test: {dopasowanie}')
print(result.params.valuesdict())

