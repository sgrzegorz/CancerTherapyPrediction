import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

params = {"P0": 1741867,
          "C0": 1,
          "lambda_p": 0.01000053,
          "gamma_p": 1.00000000,
          "KDE": 0.12759375,
          "K": 1800873.48,
          "eta": 0.01109090}



def unit_step_fun(x,threshold):
    return x*(1 / 2 + 1 / 2 *np.tanh(100 * (x-threshold)))


def f(y, t, paras):
    """
    Your system of differential equations
    """
    [P, C] = y
    try:
        lambda_p = paras['lambda_p']
        # gamma_q = paras['gamma_q'].value
        gamma_p = paras['gamma_p']
        KDE = paras['KDE']
        K = paras['K']
        eta = paras['eta']
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


t_true = np.arange(0,250)

data_fitted = g(t_true, [params["P0"],params["C0"]], params)



# plot fitted data
fig, (plt1,plt2) = plt.subplots(2,1)
plt1.plot(t_true, data_fitted[:, 0], '-', linewidth=2, color='red', label='fitted data')
plt2.plot(t_true, data_fitted[:, 1], '-', linewidth=2, color='yellow', label='fitted data')
plt2.plot(t_true, np.repeat(params['eta'],len(t_true)), '--', linewidth=1, color='black', label='eta')

plt.show()