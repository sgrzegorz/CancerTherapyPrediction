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
    """
    Your system of differential equations
    """
    [P, C] = y
    try:
        alpha = paras['alpha'].value
        lambda_p = paras['lambda_p'].value
        gamma_p = paras['gamma_p'].value
        KDE = paras['KDE'].value
        K = paras['K'].value
        eta = paras['eta'].value
        eta = eta*paras['C0'].value
    except KeyError:
        # lambda_p, gamma_q,gamma_p,KDE,k_pq,K,eta = paras
        print("Key error inna inicjalizacja")
        lambda_p, gamma_p,KDE,K,eta = paras

    dCdt =-KDE * C
    dPdt = lambda_p * P*(1-P/K)  - alpha*t - gamma_p * unit_step_fun(C,eta)  * P
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
    return (model - data).ravel()


df_true = pd.read_csv("data/klusek/patient3/stats_wszystkie_iteracje.csv")
from data.klusek.patient3.config import threatment_start, threatment_end,threatment2_start
threatment_time = threatment_end - threatment_start
df_true = df_true[(df_true['iteration'] >= threatment_start) & (df_true['iteration']<=threatment2_start)]

# threatment_end = df_true['prolif_cells'].idxmin()

# normalize prolif cells
# df_true.prolif_cells = NormalizeData(list(df_true.prolif_cells))*50

P_true = list(df_true.prolif_cells)
N_true = list(df_true.dead_cells)
t_true = list(df_true.iteration)

fig, (plt1,plt2) = plt.subplots(2,1)
fig.set_figheight(10)
plt1.plot(t_true, P_true,color='black', linewidth=1, label='model czasowo-przestrzenny 3d')

df = df_true
# df = df_true[df_true.index < (5/6*threatment_time)+threatment_start]

if df.empty:
    raise ValueError("No data provided!")


# warunek = []
# for i in list(df_true.index):
#     if i < threatment_end:
#         warunek.append(i % 3==0)
#     else:
#         warunek.append(i % 40==0)
# df = df_true[warunek]


# df = df.iloc[::1000,:] #we?? co n-ty wynik

P = list(df.prolif_cells)
N = list(df.dead_cells)
t = list(df.iteration)

plt1.scatter(t, P,color='yellow', label='przedzia?? uczenia')
plt1.set_title("Rys1 Kom??rki proliferatywne")
# plt.scatter(t, N, color='blue', label='N taken')
plt2.scatter(t, np.repeat(0,len(t)), color='yellow', label='przedzia?? uczenia')
plt2.set_title("Rys2 Lekarstwo")

# measured data
x2_measured = np.array([P]).T

# set parameters including bounds; you can also fix parameters (use vary=False)
params = Parameters()
params.add('P0', value=P[0], vary=False)
params.add('C0', min=3, max=13)
params.add('gamma_p',value=0.003, min=0.0000001, max=.1)
params.add('KDE', value=0.007, min=0.00001, max=0.7) #uwaga KDE jest modyfikowana w f,
params.add('K', value=1.9e6, min=1.8e6, max=3.e6)
# params.add('eta', expr='0.2*C0')
params.add('eta', value=0.2, min=0.1, max=0.3) #uwaga eta jest modyfikowana w f, min=0.1 bedzie min=0.1*C0
params.add('KDE', value=0.007, expr=f'-ln(eta)/({threatment_time}+200)')
params.add('alpha', min=0.000005, max=0.08)

# alpha < lambda_p bo alpha/lambda_p ma byc  u??amkiem
params.add('alpha_diff', value=15,min=15, max=25)
params.add('lambda_p', expr='alpha_diff * alpha')


# fit model
result = minimize(residual, params, args=(t, x2_measured), method='powell')  # leastsq nelder
data_fitted = g(t_true, [P[0], result.params['C0'].value], result.params)

# plot fitted data
params_eta =result.params['eta'].value*result.params['C0'].value
plt1.plot(t_true, data_fitted[:, 0], '-', linewidth=1, color='green', label='model uproszczony ')
plt1.set_ylabel("cells count")
plt1.set_xlabel("time")
plt2.plot(t_true, data_fitted[:, 1], '-', linewidth=1, color='green', label='model uproszczony')
plt2.plot(t_true, np.repeat(params_eta,len(t_true)), '--', linewidth=1, color='blue', label='threshold')
plt2.set_xlabel("time\n1)poni??ej poziomu threshold w modelu uproszczonym lekarstwo nie dzia??a")

plt2.plot(t_true,[unit_step_fun(x,params_eta) for x in data_fitted[:, 1]],'--',linewidth=1,color='brown',label="efektywno???? lekarstwa")
plt1.legend()
plt2.legend()

# display fitted statistics
report_fit(result)


plt.show()

print(result.params.valuesdict())

