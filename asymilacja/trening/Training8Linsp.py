import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint

def unit_step_fun(x,threshold):
    return x*(1 / 2 + 1 / 2 *np.tanh(100 * (x-threshold)))

# def unit_step_fun(x, threshold):
#     if x >= threshold:
#         return x
#     else:
#         return 0.0

def f(y, t, paras):
    """
    Your system of differential equations
    """
    [P, C] = y
    try:
        lambda_p = paras['lambda_p'].value
        gamma_p = paras['gamma_p'].value
        KDE = paras['KDE'].value
        K = paras['K'].value
        eta = paras['eta'].value
    except KeyError:
        # lambda_p, gamma_q,gamma_p,KDE,k_pq,K,eta = paras
        lambda_p, gamma_p,KDE,K,eta = paras


    dCdt = -KDE * C

    dPdt = lambda_p * P*(1-P/K) - gamma_p * unit_step_fun(C,eta) * KDE * P
    # dPdt = lambda_p * P*(1-P/K) - k_pq * P - gamma_p * C * KDE * P

    return [dPdt, dCdt]


def g(t, x0, paras):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    x = odeint(f, x0, t, args=(paras,))
    return x


def residual(ps, ts, data):
    x0 = ps['P0'].value,  ps['C0'].value
    model = g(ts, x0, ps)
    return (model - data).ravel()


df_true = pd.read_csv("data/klusek/patient4/okres.csv")
threatment_end = df_true['prolif_cells'].idxmin()

# fix curement function
from asymilacja.utlis.curement1 import curement_function
df_true.curement = [curement_function(i,0,threatment_end) for i in list(df_true.index)]


P_true = list(df_true.prolif_cells)
N_true = list(df_true.dead_cells)
C_true = list(df_true.curement)
t_true = list(df_true.index)

fig, (plt1,plt2) = plt.subplots(2,1)
fig.tight_layout(pad=4.0)


plt1.plot(t_true, P_true,color='black', linewidth=1, label='model Adriana')
# plt.scatter(t_true, N_true, color='green', label='N truth')
plt2.plot(t_true, C_true, color='black', linewidth=1, label='Funkcja liniowa')
# df.loc[df['prolif_cells'].idxmin()]['iteration']



df = df_true[df_true.index < threatment_end]
# df = df[df.index % 5 == 0]
P = list(df.prolif_cells)
N = list(df.dead_cells)
C = list(df.curement)
t = list(df.index)





plt1.scatter(t, P,color='yellow', label='przedział uczenia')
plt1.set_title("Rys1 Komórki proliferatywne")
# plt.scatter(t, N, color='blue', label='N taken')
plt2.scatter(t, C, color='yellow', label='przedział uczenia')
plt2.set_title("Rys2 Lekarstwo")
# initial conditions
y0 = [P[0], C[0]]



# measured data
x2_measured = np.array([P,C]).T

# set parameters including bounds; you can also fix parameters (use vary=False)
params = Parameters()
params.add('P0', value=y0[0], vary=False)
params.add('C0', value=y0[1], vary=False)
params.add('lambda_p', value=0.5, min=0.01, max=0.3)
params.add('gamma_p', value=0.3, min=0.0001, max=1.)
params.add('KDE', value=0.03, min=0.01, max=0.2)
params.add('K', value=1.9e6, min=1.8e6, max=2.e6)
params.add('eta', value=0.2, min=0.1, max=0.4)

# fit model
result = minimize(residual, params, args=(t, x2_measured), method='leastsq')  # leastsq nelder
# check results of the fit
data_fitted = g(t_true, y0, result.params)


# plot fitted data
plt1.plot(t_true, data_fitted[:, 0], '-', linewidth=1, color='green', label='model uproszczony ')
plt1.set_ylabel("cells count")
plt1.set_xlabel("time")
plt2.plot(t_true, data_fitted[:, 1], '-', linewidth=1, color='green', label='model uproszczony')
plt2.plot(t_true, np.repeat(result.params['eta'].value,len(t_true)), '--', linewidth=1, color='blue', label='threshold')
plt2.set_xlabel("time\n1)poniżej poziomu threshold w modelu uproszczonym lekarstwo nie działa \n2)Poziom lekarstwa w modelu Adriana symuluję funkcją liniową")

plt1.legend()
plt2.legend()

# plt.xlim([0, max(t)])
# plt.ylim([0, 1.1 * max(data_fitted[:, 1])])
# display fitted statistics
report_fit(result)

plt.show()