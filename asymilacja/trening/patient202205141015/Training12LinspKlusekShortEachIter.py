import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint

from asymilacja.utils.preprocessAndNormalize import NormalizeData

def plot_assimilation(t_true,P_true, P_fitted, C_fitted,params_eta,t=None,P=None,t_measured=None,P_measured=None):
    fig, (plt1,plt2) = plt.subplots(2,1)
    fig.set_figheight(10)
    if t is not None and P is not None:
        plt1.scatter(t, P,color='yellow', label='przedział uczenia')
        plt2.scatter(t, np.repeat(0, len(t)), color='yellow', label='przedział uczenia')

    if t_measured is not None and P_measured is not None:
        plt1.scatter(t_measured, P_measured,color='red', label='pomiary wykorzystane do uczenia')

    plt1.plot(t_true, P_true,color='black', linewidth=1, label='model Adriana')
    plt1.set_title("Rys1 Komórki proliferatywne")
    plt1.plot(t_true, P_fitted, '-', linewidth=1, color='green', label='model uproszczony ')
    plt1.set_ylabel("cells count")
    plt1.set_xlabel("time")
    plt1.legend()

    plt2.set_title("Rys2 Lekarstwo")
    plt2.plot(t_true, C_fitted, '-', linewidth=1, color='green', label='model uproszczony')
    plt2.plot(t_true, np.repeat(params_eta,len(t_true)), '--', linewidth=1, color='blue', label='threshold')
    plt2.set_xlabel("time\n1)poniżej poziomu threshold w modelu uproszczonym lekarstwo nie działa")
    plt2.plot(t_true,[unit_step_fun(x,params_eta) for x in data_fitted[:, 1]],'--',linewidth=1,color='brown',label="efektywność lekarstwa")
    plt2.legend()

    plt.show()


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


df_true = pd.read_csv("data/klusek/patient202205141015/stats0.csv")
from data.klusek.patient202205141015.config import threatment_start, threatment_end,threatment2_start
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
N = list(df.dead_cells)
t = list(df.iteration)

params = Parameters()
params.add('P0', value=P[0], vary=False)
params.add('C0', min=1e-5, max=10)
params.add('gamma_p', min=0.00005, max=0.2)
params.add('K', value=0.35e6, min=0.3e6, max=0.5e6)
params.add('T_death', value=0.35e6, min=0.3e6, max=0.5e8)
params.add('eta', value=0.1, min=0.05, max=0.2) #uwaga eta jest modyfikowana w f, min=0.1 bedzie min=0.1*C0
params.add('KDE',  expr=f'-ln(eta)/({threatment_time}+200)')
params.add('lambda_p', min=0.00005, max=0.002)

df_1 = df.loc[df['iteration'] <= threatment_end]
df_2 = df.loc[df['iteration'] > threatment_end]
df_1 = df_1.iloc[::200,:]
df_2 = df_2.iloc[::2000,:]
df_sampled = pd.concat([df_1,df_2])
t_measured = list(df_sampled.iteration)
P_measured = list(df_sampled.prolif_cells)
# t_measured = np.linspace(threatment_start,threatment_end,num=4)
# t_measured = np.concatenate((t_measured,np.linspace(threatment_end,threatment2_start,num=6)))
# t_measured = np.around(t_measured)
# P_measured = df.loc[t_measured,'prolif_cells'].to_list()


result = minimize(residual, params, args=(t_measured, P_measured), method='least_squares')  # leastsq nelder

x0 = [P[0], result.params['C0'].value]
data_fitted = g(t_true, x0, result.params)
P_fitted = data_fitted[:, 0]
C_fitted =data_fitted[:, 1]
params_eta =result.params['eta'].value*result.params['C0'].value
plot_assimilation(t_true,P_true, P_fitted, C_fitted,params_eta,t=t,P=P,t_measured=t_measured, P_measured=P_measured)


report_fit(result)
print(result.params.valuesdict())
