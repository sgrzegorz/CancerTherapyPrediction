import pyabc
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint, solve_ivp
import tempfile
import os
import pandas as pd
from asymilacja.model.Cancer5KlusekShort import CancerModel
from asymilacja.paramteres.Vis2KlusekShort import plot_simple_results, plot_gt

df = pd.read_csv("data/klusek/patient4/okres.csv")
plot_gt("data/klusek/patient4/okres.csv")
df = df[df.index % 20 == 0]
# df = df[df.index <df.shape[0]/10]
# df = df[df.index <3*df.shape[0]/10]
# df = df[df.index <5*df.shape[0]/10]
# df = df[df.index <6*df.shape[0]/10]



P = list(df.prolif_cells)
N = list(df.dead_cells)
C = list(df.curement)
C = list(map(lambda x: x*10**6,C))
observation = np.array([P, N, C], dtype=float)

t_span = np.array([list(df.t)[0], list(df.t)[-1]])
t = list(df.t)


def model(parameters):
    global P, N, t, C

    lambda_p = parameters["lambda_p"]
    gamma_q = parameters["gamma_q"]
    gamma_p = parameters["gamma_p"]
    KDE = 0.01
    k_pq = parameters["k_pq"]
    # K = parameters["K"]
    K = -1
    eta = parameters["eta"]
    # C0 = C[0]
    y0 = [P[0], N[0], C[0]]

    m1 = CancerModel(lambda_p, gamma_q, gamma_p, KDE, k_pq, K, eta)
    #    soln = solve_ivp(m1.model, t_span, y0, t_eval=times,method='BDF',rtol=1e9,atol=1e9)
    sol = solve_ivp(m1.model, t_span, y0, t_eval=t, method='Radau',rtol=1e9, atol=1e9)

    if not sol.success:
        print(sol.message)
    solt = sol.t
    solP = sol.y[0]
    solN = sol.y[1]
    solC = sol.y[2]

    return {"data": np.array([solP, solN, solC], dtype=float)}


def distance(x, y):
    return np.absolute(x["data"][0] - y["data"][0]).sum()


def distance1(x, y):
    return np.absolute(x["data"][1] - y["data"][1]).sum()

def distance2(x, y):
    return np.absolute(x["data"][2] - y["data"][2]).sum()

dist = pyabc.distance.AggregatedDistance([distance, distance1,distance2])

prior = pyabc.Distribution(
    KDE=pyabc.RV("uniform", 0.01, 1), k_pq=pyabc.RV("uniform", 0.01, 1),  # K=pyabc.RV("uniform", 50, 250), C0=pyabc.RV("uniform", 0.01, 10)
    lambda_p=pyabc.RV("uniform", 0.01, 1), gamma_p=pyabc.RV("uniform", 0.01, 1),
    gamma_q=pyabc.RV("uniform", 0.01, 1), eta=pyabc.RV("uniform", 0.01, 1))

abc = pyabc.ABCSMC(model, prior, dist, population_size=100)

db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db"))

abc.new(db_path, {"data": observation})

history = abc.run(minimum_epsilon=10, max_nr_populations=100)

history is abc.history
run_id = history.id

posterior2 = pyabc.MultivariateNormalTransition()
posterior2.fit(*history.get_distribution(m=0))

t_params = posterior2.rvs()

print(t_params)

for i in range(50):
    abc_continued = pyabc.ABCSMC(model, prior, distance)
    abc_continued.load(db_path, run_id)
    history = abc_continued.run(minimum_epsilon=.1, max_nr_populations=4)

    posterior2 = pyabc.MultivariateNormalTransition()
    posterior2.fit(*history.get_distribution(m=0))
    t_params = posterior2.rvs()
    print(t_params)

    data = model(t_params)["data"]
    solP = data[0, :]
    solN = data[1, :]
    solC = data[2, :]
    plt.plot(t, solC, label='C')
    plt.title('Assimilated Curement')
    plt.show()
    plt.scatter(t, solP, label='P')
    plt.scatter(t, solN, label='N')
    plt.title("Assimilated data")
    plt.legend()
    plt.show()
