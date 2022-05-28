import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import odeint



def unit_step_fun(x,threshold):
    return x*(1 / 2 + 1 / 2 *np.tanh(100 * (x-threshold)))

def feed_forward(y,t,paras):
    [P, C] = y
    alpha = paras['alpha']
    lambda_p = paras['lambda_p']
    gamma_p = paras['gamma_p']
    KDE = paras['KDE']
    K = paras['K']
    eta = paras['eta']
    eta = eta * paras['C0']

    dCdt = -KDE * C
    dPdt = lambda_p * P * (1 - P / K) - alpha * t - gamma_p * unit_step_fun(C, eta) * P
    return [dPdt, dCdt]


def plot_parameters(differentialForward,parameters,steps_forward,steps_backward,threatment_start,label=None,USE_REAL_TIME=False,t_real=None,differentialBackward=None,lineLabel=None):
    if differentialBackward is None:
        differentialBackward = differentialForward
    # steps_backward=1000
    t1 = np.linspace(0, -steps_backward, steps_backward + 1)+threatment_start
    x0 = [parameters['P0'],0.0]
    x = odeint(differentialBackward, x0, t1,args=(parameters,))
    P1 = x[:, 0]
    C1 = x[:, 1]

    # steps_forward=250
    t2 = np.linspace(0, steps_forward, steps_forward + 1)+threatment_start
    x0 = [parameters['P0'],parameters['C0']]
    x = odeint(differentialForward, x0, t2,args=(parameters,))
    P2 = x[:, 0]
    C2 = x[:, 1]

    # wartosc poczatku terapii powtarza sie w t1 i t2, wiec z jednego musimy ja pominąc
    t2 =t2[1:]
    P2 =P2[1:]


    t= np.concatenate((np.flip(t1),t2))
    P = np.concatenate((np.flip(P1),P2))


    if USE_REAL_TIME:
        plt.plot(t_real, P, label=label)
        if lineLabel is not None:
            plt.text(t_real[-1], P[-1], f'{lineLabel}')
    else:
        plt.plot(t, P, label=label)
        if lineLabel is not None:
            plt.text(t[-1], P[-1], f'{lineLabel}')
    plt.ylabel("liczba komórek nowotworowych")

def plot_truth(t_true,P_true,USE_REAL_TIME=False,t_real=None):
    if USE_REAL_TIME:
        plt.plot(t_real, P_true, color='black', linewidth=2, label='model czasowo-przestrzenny 3d')
        plt.xlabel('time [days]')
    else:
        plt.plot(t_true, P_true,color='black', linewidth=2, label='model czasowo-przestrzenny 3d')
        plt.xlabel('numer iteracji')

def plot_curement(differentialMethod,parameters,steps_forward,threatment_start,params_eta=None):
    if params_eta is None:
        params_eta = parameters['eta']

    t2 = np.linspace(0, steps_forward, steps_forward + 1) + threatment_start
    x0 = [parameters['P0'], parameters['C0']]
    x = odeint(differentialMethod, x0, t2, args=(parameters,))
    C_fitted = x[:, 1]
    plt.title("Rys2 Lekarstwo")
    plt.plot(t2, C_fitted, '-', linewidth=1, color='green', label='model uproszczony')
    plt.plot(t2, np.repeat(params_eta, len(t2)), '--', linewidth=1, color='blue', label='threshold')
    plt.xlabel("time\n1)poniżej poziomu threshold w modelu uproszczonym lekarstwo nie działa")
    plt.plot(t2, [unit_step_fun(x, params_eta) for x in C_fitted], '--', linewidth=1, color='brown',
              label="efektywność lekarstwa")
    # plt.legend()
    plt.xlim([0, t2[-1]])

    # plt.show()


if __name__ == '__main__':
    from data.klusek.patient3.config import threatment_start, threatment_end,threatment2_start

    USE_REAL_TIME = True

    threatment_time = threatment_end - threatment_start
    steps_backward = threatment_start
    steps_forward = threatment2_start - threatment_start

    df = pd.read_csv("data/klusek/patient3/stats_wszystkie_iteracje.csv")
    df_true = df[(df['iteration'] >= 0) & (df['iteration']<=threatment2_start)]
    P_true = list(df_true.prolif_cells)
    t_true = list(df_true.iteration)
    t_real = list(df_true.t)

    threesixth = {'P0': 1578183, 'C0': 3.046479523664936, 'gamma_p': 0.00045138352522655937, 'KDE': 0.00044991509877651383, 'K': 2999999.999749261, 'eta': 0.2999999999999955, 'alpha': 5e-06, 'alpha_diff': 15.000000000000611, 'lambda_p': 7.500000000000306e-05}

    fivesixth = {'P0': 1578183, 'C0': 3.0003485280513806, 'gamma_p': 0.0005424607534442532, 'KDE': 0.0004499150987765088, 'K': 2999999.999999944, 'eta': 0.2999999999999996, 'alpha': 5e-06, 'alpha_diff': 15.000000000000231, 'lambda_p': 7.500000000000116e-05}

    sixsixth = {'P0': 1578183, 'C0': 3.000224519429239, 'gamma_p': 0.000558085709592946, 'KDE': 0.0004499150987765093, 'K': 2999999.9999999944, 'eta': 0.29999999999999916, 'alpha': 5.000000000004441e-06, 'alpha_diff': 15.000000000002899, 'lambda_p': 7.500000000008112e-05}

    fullData ={'P0': 1578183, 'C0': 3.0001046341507203, 'gamma_p': 0.0006277530244156134, 'KDE': 0.0005304557712284162, 'K': 1801293.3100846584, 'eta': 0.24183499026463035, 'alpha': 5.256099560277746e-06, 'alpha_diff': 16.625711306862783, 'lambda_p': 8.738639388930623e-05}

    plot_truth(t_true, P_true, USE_REAL_TIME,t_real)

    plot_parameters(feed_forward,fullData,steps_forward,steps_backward,threatment_start,"fullData",USE_REAL_TIME,t_real)
    plot_parameters(feed_forward,threesixth,steps_forward,steps_backward,threatment_start,"3/6 threatmentTime",USE_REAL_TIME,t_real)
    plot_parameters(feed_forward,fivesixth,steps_forward,steps_backward,threatment_start,"5/6 threatmentTime",USE_REAL_TIME,t_real,lineLabel="5/6")
    plot_parameters(feed_forward,sixsixth,steps_forward,steps_backward,threatment_start,"6/6 threatmentTime",USE_REAL_TIME,t_real,lineLabel="full")


    plt.legend()
    plt.show()
