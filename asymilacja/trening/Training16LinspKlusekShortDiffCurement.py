import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint

from asymilacja.utils.preprocessAndNormalize import NormalizeData

def plot_assimilation(t_true,P_true, P_fitted, C_fitted,params_eta,t=None,P=None):
    fig, (plt1,plt2) = plt.subplots(2,1)
    fig.set_figheight(10)
    if t is not None and P is not None:
        plt1.scatter(t, P,color='yellow', label='przedział uczenia')
        plt2.scatter(t, np.repeat(0, len(t)), color='yellow', label='przedział uczenia')

    plt1.plot(t_true, P_true,color='black', linewidth=1, label='model Adriana')
    plt1.set_title("Rys1 Komórki proliferatywne")
    plt1.plot(t_true, P_fitted, '-', linewidth=1, color='green', label='model uproszczony ')
    plt1.set_ylabel("cells count")
    plt1.set_xlabel("time")
    plt1.legend()

    plt2.set_title("Rys2 Lekarstwo")
    plt2.plot(t_true, C_fitted, '-', linewidth=1, color='green', label='model uproszczony')
    plt2.plot(t_true, np.repeat(params_eta,len(t_true)), '--', linewidth=1, color='blue', label='threshold (parametr eta)')
    plt2.set_xlabel("time\n1)poniżej poziomu threshold w modelu uproszczonym lekarstwo nie działa")
    plt2.plot(t_true,[unit_step_fun(x,params_eta) for x in data_fitted[:, 1]],'--',linewidth=1,color='brown',label="efektywność lekarstwa")
    plt2.legend()

    plt.show()

def unit_step_fun(x,threshold):
    return x*(1 / 2 + 1 / 2 *np.tanh(100 * (x-threshold)))

def f(y, t, paras):
    """
    Your system of differential equations
    """
    [P, C] = y
    try:
        lambda_p = paras['lambda_p'].value
        gamma_p = paras['gamma_p'].value
        KDE = paras['KDE'].value
        KDE2 = paras['KDE2'].value
        K = paras['K'].value
        eta = paras['eta'].value
        eta = eta*paras['C0'].value
    except KeyError:
        # lambda_p, gamma_q,gamma_p,KDE,k_pq,K,eta = paras
        print("Key error inna inicjalizacja")
        lambda_p, gamma_p,KDE,K,eta = paras

    dCdt =-KDE * C*P -KDE2*C
    dPdt = lambda_p * P*(1-P/K)  - gamma_p * unit_step_fun(C,eta)  * P
    return [dPdt, dCdt]


def g(t, x0, paras):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    x = odeint(f, x0, t, args=(paras,))
    return x


def residual(ps, ts, data):
    x0 = ps['P0'].value, ps['C0'].value
    model = g(ts, x0, ps)
    return (model[:,0] - data).ravel()


df_true = pd.read_csv("data/klusek/patient202205141015/stats0.csv")
from data.klusek.patient202205141015.config import threatment_start, threatment_end,threatment2_start
threatment_time = threatment_end - threatment_start
df_true = df_true[(df_true['iteration'] >= threatment_start) & (df_true['iteration']<=threatment2_start)]

# threatment_end = df_true['prolif_cells'].idxmin()

# normalize prolif cells
# df_true.prolif_cells = NormalizeData(list(df_true.prolif_cells))*50

P_true = list(df_true.prolif_cells)
N_true = list(df_true.dead_cells)
t_true = list(df_true.iteration)


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


# df = df.iloc[::10,:] #weź co n-ty wynik

P = list(df.prolif_cells)
t = list(df.iteration)


# measured data
x2_measured = np.array([P]).T
maxi = np.max(df.prolif_cells)

# set parameters including bounds; you can also fix parameters (use vary=False)
params = Parameters()
params.add('P0', value=P[0], vary=False)
params.add('C0', min=3, max=10)
params.add('gamma_p',value=0.003, min=0.0001, max=0.01)
params.add('K', value=maxi, min=0.9*maxi, max=1.1*maxi)
# params.add('eta', expr='0.2*C0')
params.add('eta', value=0.2, min=0.1, max=0.3) #uwaga eta jest modyfikowana w f, min=0.1 bedzie min=0.1*C0
params.add('KDE', value=0.007, min=1e-6,max=1.5e-6)
params.add('KDE2', value=0.007, min=0.0001,max=0.01)
params.add('lambda_p', value=0.005,min=0.0001,max=0.01)

t_measured = np.linspace(threatment_start,threatment_end,num=4)
# t_measured=t_measured[:-1]
t_measured = np.concatenate((t_measured,np.linspace(threatment_end,threatment2_start,num=6)))
t_measured = np.around(t_measured)
# t = [int(i) for i in t]
P_measured = df.loc[t_measured,'prolif_cells'].to_list()
# fit model
result = minimize(residual, params, args=(t_measured, P_measured), method='powell')  # leastsq nelder

x0 = [P[0], result.params['C0'].value]

data_fitted = g(t_true, x0, result.params)
P_fitted = data_fitted[:, 0]
C_fitted =data_fitted[:, 1]
params_eta =result.params['eta'].value*result.params['C0'].value

plot_assimilation(t_true,P_true, P_fitted, C_fitted,params_eta,t=t,P=P)

# display fitted statistics
report_fit(result)

print(result.params.valuesdict())

