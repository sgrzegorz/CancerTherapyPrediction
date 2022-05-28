import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint


def plot_assimilation(t_true,P_true, P_fitted, C_fitted,params_eta,t=None,P=None,t_measured=None,P_measured=None):
    fig, (plt1,plt2) = plt.subplots(2,1)
    fig.set_figheight(10)
    if t is not None and P is not None:
        plt1.scatter(t, P,color='yellow', label='przedział uczenia')
        plt2.scatter(t, np.repeat(0, len(t)), color='yellow', label='przedział uczenia')

    if t_measured is not None and P_measured is not None:
        plt1.scatter(t_measured, P_measured,color='red', label='pomiary wykorzystane do uczenia')

    plt1.plot(t_true, P_true,color='black', linewidth=1, label='model czasowo-przestrzenny 3d')
    plt1.set_title("Rys1 Komórki proliferatywne")
    plt1.plot(t_true, P_fitted, '-', linewidth=1, color='green', label='model uproszczony ')
    plt1.set_ylabel("cells count")
    plt1.set_xlabel("time")
    plt1.legend()

    plt2.set_title("Rys2 Lekarstwo")
    plt2.plot(t_true, C_fitted, '-', linewidth=1, color='green', label='model uproszczony')
    plt2.plot(t_true, np.repeat(params_eta,len(t_true)), '--', linewidth=1, color='blue', label='threshold')
    plt2.set_xlabel("time\n1)poniżej poziomu threshold w modelu uproszczonym lekarstwo nie działa")
    plt2.plot(t_true,[unit_step_fun(x,params_eta) for x in C_fitted],'--',linewidth=1,color='brown',label="efektywność lekarstwa")
    plt2.legend()

    plt.show()


def unit_step_fun(x,threshold):
    return x*(1 / 2 + 1 / 2 *np.tanh(100 * (x-threshold)))

