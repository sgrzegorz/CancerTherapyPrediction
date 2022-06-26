import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint

from asymilacja.utils.preprocessAndNormalize import NormalizeData
from asymilacja.trening.utils import *


def f(y, t, paras):
    [P, C] = y
    alpha = paras['alpha'].value
    lambda_p = paras['lambda_p'].value
    gamma_p = paras['gamma_p'].value
    KDE = paras['KDE'].value
    K = paras['K'].value
    eta = paras['eta'].value
    eta = eta*paras['C0'].value

    dCdt =-KDE * C
    dPdt = lambda_p * P*(1-P/K)  - alpha*t - gamma_p * unit_step_fun(C,eta)  * P
    return [dPdt, dCdt]

def g(t, x0, paras):
    x, info  = odeint(f, x0, t, args=(paras,),full_output=True)
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
# df = df_true[df_true.index < (5/6*threatment_time)+threatment_start]

if df.empty:
    raise ValueError("No data provided!")

P = list(df.prolif_cells)
N = list(df.dead_cells)
t = list(df.iteration)

maxi = np.max(df.prolif_cells)
params = Parameters()
params.add('P0', value=P[0], vary=False)
params.add('C0', min=3, max=10)
params.add('gamma_p',value=0.003, min=0.0000001, max=.1)
params.add('KDE', value=0.007, min=0.00001, max=0.7) #uwaga KDE jest modyfikowana w f,
params.add('K', value=1.1 * maxi, min=1.0 * maxi, max=1.7 * maxi)
params.add('eta', value=0.2, min=0.1, max=0.3) #uwaga eta jest modyfikowana w f, min=0.1 bedzie min=0.1*C0
params.add('KDE', value=0.007, expr=f'-ln(eta)/({threatment_time}+200)')
params.add('alpha', min=0.000005, max=0.08)
# alpha < lambda_p bo alpha/lambda_p ma byc  ułamkiem
params.add('alpha_diff', value=15,min=15, max=25)
params.add('lambda_p', expr='alpha_diff * alpha')
assimilated_parameters ={'P0': 587182, 'C0': 3.2612874368754086, 'gamma_p': 0.0003929552085542292, 'KDE': 0.0003944865044121891, 'K': 2239789.0593792014, 'eta': 0.2999999978579805, 'alpha': 5.000000014716197e-06, 'alpha_diff': 15.000000000053898, 'lambda_p': 7.500000022101244e-05}
for parName, parVal in assimilated_parameters.items():
    params[parName].set(value=parVal)

df_1 = df.loc[df['iteration'] <= threatment_end]
df_2 = df.loc[df['iteration'] > threatment_end]
df_1 = df_1.iloc[::200,:]
df_2 = df_2.iloc[::2000,:]
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

report_fit(result)
dopasowanie = np.linalg.norm(P_fitted - P_true, ord=2)
print(f'Least square test: {dopasowanie}')
print(result.params.valuesdict())

