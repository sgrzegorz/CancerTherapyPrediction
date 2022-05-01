import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import odeint


def unit_step_fun(x,threshold):
    return x*(1 / 2 + 1 / 2 *np.tanh(100 * (x-threshold)))


def feed_backward(y,t,paras):
    [P, C] = y
    lambda_p = paras['lambda_p']
    psi_p = paras['psi_p']
    gamma_p = paras['gamma_p']
    K = paras['K']
    KDE = paras['KDE']
    eta = paras['eta']

    KDE = KDE * paras['C0']
    eta = eta*paras['C0']

    dCdt =-KDE * C
    dPdt = lambda_p * P- psi_p*P- gamma_p * unit_step_fun(C,eta)  * P
    # print(lambda_p * P*(1-P/K),(1-P/K),psi_p*P,P)
    return [dPdt, dCdt]

def feed_forward(y,t,paras):

    [P, C] = y
    lambda_p = paras['lambda_p']
    psi_p = paras['psi_p']
    gamma_p = paras['gamma_p']
    K = paras['K']
    KDE = paras['KDE']
    eta = paras['eta']

    KDE = KDE * paras['C0']
    eta = eta*paras['C0']

    dCdt =-KDE * C
    dPdt = lambda_p * P*(1-P/K)- psi_p*P- gamma_p * unit_step_fun(C,eta)  * P
    return [dPdt, dCdt]

def plot_parameters(parameters,label=None,):
    steps_backward=100
    t1 = np.linspace(0, -steps_backward, steps_backward + 1)
    x0 = [parameters['P0'],0.0]
    x = odeint(feed_backward, x0, t1,args=(parameters,))
    P1 = x[:, 0]
    C1 = x[:, 1]

    steps_forward=250
    t2 = np.linspace(0, steps_forward, steps_forward + 1)
    x0 = [parameters['P0'],parameters['C0']]
    x = odeint(feed_forward, x0, t2,args=(parameters,))
    P2 = x[:, 0]
    C2 = x[:, 1]

    t= np.concatenate((np.flip(t1),t2))
    P = np.concatenate((np.flip(P1),P2))

    plt.plot(t, P, label=label)


threesixth = {'P0': 1741867, 'C0': 0.3027635766261936, 'gamma_p': 0.20404838186665247, 'KDE': 0.06935515209593586, 'K': 1900000.0653516473, 'eta': 0.2, 'psi_p': 0.05012194261577609, 'lambda_p_m': 0.09667716684486889, 'lambda_p': 0.14679910946064498}

fivesixth = {'P0': 1741867, 'C0': 0.30000000008297634, 'gamma_p': 0.25389663238988214, 'KDE': 0.06779097962044832, 'K': 1900000.0338206063, 'eta': 0.2, 'psi_p': 0.05000000005593625, 'lambda_p_m': 0.06860679348824308, 'lambda_p': 0.11860679354417933}

sixsixth = {'P0': 1741867, 'C0': 0.3000000000849464, 'gamma_p': 0.2638415182169818, 'KDE': 0.06772912625477959, 'K': 1900000.0284421386, 'eta': 0.2, 'psi_p': 0.05000000006041046, 'lambda_p_m': 0.06310224987705068, 'lambda_p': 0.11310224993746114}


fullData = {'P0': 1741867, 'C0': 0.30000000000000004, 'gamma_p': 0.26068553617550844, 'KDE': 0.05000000000004916, 'K': 1807530.2223988457, 'eta': 0.10000000000000024, 'psi_p': 0.05714603743116588, 'lambda_p_m': 0.09027214899709243, 'lambda_p': 0.14741818642825832}

from asymilacja.model.Cancer6LinspKlusekShort import plot_parameters

plot_parameters(fullData,label="fullData")
plot_parameters(threesixth,label="1/2 threatmentTime")
plot_parameters(fivesixth,label="5/6 threatmentTime")
plot_parameters(sixsixth,label="threatmentTime")


df_true = pd.read_csv("data/klusek/out/2dawki_first_curement.txt",index_col=0)
P_true = list(df_true.prolif_cells)
t_true = list(df_true.index)

plt.plot(t_true, P_true,color='black', linewidth=2, label='model Adriana')


plot_parameters(fullData)
plt.legend()
plt.show()
