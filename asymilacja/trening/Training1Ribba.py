import numpy as np
import pyabc

from scipy.integrate import odeint
import tempfile
import os
import matplotlib.pyplot as plt
# %matplotlib inline
from asymilacja.utlis.datasets import ribba_dataset
from asymilacja.utlis.curement import curement_function
# pyabc.settings.set_figure_params('pyabc')  # for beautified plots
from asymilacja.model.Cancer1Ribba import CancerModel
from numpy.polynomial import Chebyshev


df = ribba_dataset('data/ribba/fig4.csv')
df = df[df["t"]>0]
threatment_start = 0
threatment_end = 26
df['curement'] = [curement_function(i,threatment_start,threatment_end) for i in df.t.tolist()]

start = df.t.tolist()[0]
end = df.t.tolist()[-1]
t = np.arange(start,end)
# plt.plot(df)
# plt.show()
fit = Chebyshev.fit(df.t.tolist(), df.mtd.tolist(), deg=6)
mtd = [fit(i) for i in t]
curement = [curement_function(i,threatment_start,threatment_end) for i in t]




# plt.scatter(t,mtd)
# plt.xlabel("time [months]")
# plt.ylabel("mtd [mm]")
# plt.show()
# plt.plot(df.t,df['curement'])
# plt.show()


P0 = 0.1 *mtd[0]
Q0 = mtd[0] - P0
Q_p0 = 0.0
C0 = curement[0]
observation = np.array([ mtd,curement ])

count = len(t)

X0 = [P0, Q0, Q_p0, C0]

def model(parameters):
    # X0 = [parameters["P"], parameters["Q"], parameters["Q_p"], parameters["C"]]

    lambda_p = parameters["lambda_p"]
    delta_qp = parameters["delta_qp"]
    gamma_q = parameters["gamma_q"]
    gamma_p = parameters["gamma_p"]
    KDE = parameters["KDE"]
    k_qpp = parameters["k_qpp"]
    k_pq = parameters["k_pq"]
    K = parameters["K"]


    m1 = CancerModel(lambda_p, delta_qp, gamma_q, gamma_p, KDE, k_qpp, k_pq, K)
    # t = m1.time_interval(0,200)

    x = odeint(m1.model, X0, t)

    P = x[:, 0]
    Q = x[:, 1]
    Q_p = x[:, 2]
    C = x[:, 3]

    return {"data" :np.array([ P+Q+Q_p,C],dtype=float)}


def distance1(x, y):
    dif = 0
    for i in range(count):
        dif += abs(x["data"][0][i] - y["data"][0][i])
    return dif

def distance2(x, y):
    return np.absolute(x["data"][1]- y["data"][1]).sum()

dist = pyabc.distance.AggregatedDistance([distance1,distance2])
prior = pyabc.Distribution(lambda_p=pyabc.RV("uniform", 1, 20), delta_qp=pyabc.RV("uniform",0, 20), gamma_q=pyabc.RV("uniform", 0, 20), gamma_p=pyabc.RV("uniform", 0.1, 20), KDE=pyabc.RV("uniform", 0.01, 20), k_qpp=pyabc.RV("uniform", 0.01, 20),
                           k_pq=pyabc.RV("uniform", 0.1, 20), K=pyabc.RV("uniform", 50.0, 200))

abc = pyabc.ABCSMC(model, prior, dist, population_size=100)

db_path = ("sqlite:///" +  os.path.join(tempfile.gettempdir(), "test.db"))

abc.new(db_path, {"data": observation})

history = abc.run(minimum_epsilon=100, max_nr_populations=100)

history is abc.history

posterior2 = pyabc.MultivariateNormalTransition()
posterior2.fit(*history.get_distribution(m=0))

t_params = posterior2.rvs()

print(t_params)

data = model(t_params)["data"]
mtd = data[0,:]
# t = data[0,:]

plt.title("out")
plt.plot(t,mtd)
plt.show()

