import pyabc

from scipy.integrate import odeint
import tempfile
import os
import matplotlib.pyplot as plt
# %matplotlib inline
from asymilacja.utlis.datasets import ribba_dataset

# pyabc.settings.set_figure_params('pyabc')  # for beautified plots
from asymilacja.model.Cancer1Ribba import CancerModel

# data = pd.read_csv('../data/Zakazenia30323112020.csv', sep=';', encoding='windows-1250')
# data2 = data.copy()
# data2["Date_reported"] = dt.strptime(data["Data"][0], "%d.%m.%Y").date()
# for i in range(data["Data"].size):
#     data2["Date_reported"][i] = dt.strptime(data["Data"][i], "%d.%m.%Y").date()
# data_filtered = data2[(data2["Date_reported"]>=dt(2020, 9, 16).date()) & (data2["Date_reported"]<=dt(2020, 10, 31).date())]

#

patient = ribba_dataset('data/ribba/fig4.csv')
patient = patient[patient["t"]>0]

plt.scatter(patient.t,patient.mtd)
plt.xlabel("time [months]")
plt.ylabel("mtd [mm]")

P = 0.1 *patient.mtd[0]
Q = patient.mtd[0] - P
Q_p = 0.0
C = 0.0
observation = {"t" : patient["t"].tolist(), "mtd" : patient["mtd"].tolist()}

count = len(observation["t"])


def model(parameters):
    X0 = [parameters["P"], parameters["Q"], parameters["Q_p"], parameters["C"]]

    lambda_p = parameters["lambda_p"]
    delta_qp = parameters["delta_qp"]
    gamma_q = parameters["gamma_q"]
    gamma_p = parameters["gamma_p"]
    KDE = parameters["KDE"]
    k_qpp = parameters["k_qpp"]
    k_pq = parameters["k_pq"]
    K = parameters["K"]


    m1 = CancerModel(lambda_p, delta_qp, gamma_q, gamma_p, KDE, k_qpp, k_pq, K)
    t = m1.time_interval(0,200)

    x = odeint(m1.model, X0, t)

    P = x[:, 0]
    Q = x[:, 1]
    Q_p = x[:, 2]
    C = x[:, 3]

    return {"P" : P, "Q" : Q, "Q_p" : Q_p,"C": C}


def distance(x, y):
    dif = 0
    for i in range(count):
        dif += abs(x["data"][i] - y["data"][i])
    return dif


prior = pyabc.Distribution(lambda_p=pyabc.RV("uniform", 1, 20), delta_qp=pyabc.RV("uniform", 1.00e-8, 3.00e-8), gamma_q=pyabc.RV("uniform", 1.0e-7, 3.0e-7), gamma_p=pyabc.RV("uniform", 0.1, 1), KDE=pyabc.RV("uniform", 0.01, 0.5), k_qpp=pyabc.RV("uniform", 0.01, 0.5),
                           k_pq=pyabc.RV("uniform", 0.1, 0.7), K=pyabc.RV("uniform", 0.01, 0.5))

abc = pyabc.ABCSMC(model, prior, distance, population_size=4)

db_path = ("sqlite:///" +  os.path.join(tempfile.gettempdir(), "test.db"))

abc.new(db_path, {"data": observation})

history = abc.run(minimum_epsilon=100, max_nr_populations=4)

history is abc.history
