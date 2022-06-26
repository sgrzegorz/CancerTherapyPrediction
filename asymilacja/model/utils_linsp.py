import numpy as np


def unit_step_fun(x,threshold):
    return x*(1 / 2 + 1 / 2 *np.tanh(100 * (x-threshold)))

def differentialEquation(y, t, paras):
    [P, C] = y
    alpha = paras['alpha']
    lambda_p = paras['lambda_p']
    gamma_p = paras['gamma_p']
    KDE = paras['KDE']
    K = paras['K']
    eta = paras['eta']
    eta = eta*paras['C0']
    dCdt =-KDE * C
    dPdt = lambda_p * P*(1-P/K)  - alpha*t - gamma_p * unit_step_fun(C,eta)  * P
    return [dPdt, dCdt]

def differentialEquation_15(y, t, paras):
    [P, C] = y
    lambda_p = paras['lambda_p']
    gamma_p = paras['gamma_p']
    KDE = paras['KDE']
    K = paras['K']
    eta = paras['eta']
    eta = eta*paras['C0']
    dCdt =-KDE * C
    dPdt = lambda_p * P*(1-P/K)  - gamma_p * unit_step_fun(C,eta)  * P
    return [dPdt, dCdt]

def GS1(y, t, paras):
    [P, C] = y
    lambda_p = paras['lambda_p']
    gamma_p = paras['gamma_p']
    KDE = paras['KDE']
    K = paras['K']
    eta = paras['eta']
    eta = eta*paras['C0']

    dCdt =-KDE * C
    dPdt = lambda_p * P*(1-P/K)  - gamma_p * unit_step_fun(C,eta)  * P
    return [dPdt, dCdt]

def GS3(y, t, paras):
    [P, C] = y
    lambda_p = paras['lambda_p']
    gamma_p = paras['gamma_p']
    KDE = paras['KDE']
    K = paras['K']
    eta = paras['eta']*paras['C0']
    T_death = paras['T_death']
    dCdt =-KDE * C
    dPdt = lambda_p * P*(1-P/K) - 1/T_death*P - gamma_p * unit_step_fun(C,eta)  * P
    return [dPdt, dCdt]


def GS4(y, t, paras):
    [P, C] = y
    lambda_p = paras['lambda_p']
    gamma_p = paras['gamma_p']
    KDE = paras['KDE']
    KDE2 = paras['KDE2']
    K = paras['K']
    eta = paras['eta']
    eta = eta*paras['C0']

    dCdt =-KDE * C*P -KDE2*C
    dPdt = lambda_p * P*(1-P/K)  - gamma_p * unit_step_fun(C,eta)  * P
    return [dPdt, dCdt]

def GS5(y, t, paras):
    [P, C] = y
    alpha = paras['alpha']
    lambda_p = paras['lambda_p']
    gamma_p = paras['gamma_p']
    KDE = paras['KDE']
    K = paras['K']
    eta = paras['eta']
    eta = eta*paras['C0']

    dCdt =-KDE * C
    dPdt = lambda_p * P*(1-P/K)  - alpha*t - gamma_p * unit_step_fun(C,eta)  * P
    return [dPdt, dCdt]

