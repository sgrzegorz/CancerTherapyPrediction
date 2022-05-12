pred = {}

import numpy as np
import pyabc

from scipy.integrate import odeint
import tempfile
import os
import matplotlib.pyplot as plt
from asymilacja.utils.datasets import ribba_dataset
from asymilacja.utils.curement import curement_function
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
    weight1 = parameters["weight1"]
    weight2 = parameters["weight2"]
    weight3 = parameters["weight3"]

    pred = {'K': 196.91524201781607, 'KDE': 0.02993977035657992, 'delta_qp': 0.518550701246765,'gamma_p': 14.179606546920718, 'gamma_q': 2.85767841085851, 'k_pq': 0.6451121356194429, 'k_qpp': 0.010125204708275796, 'lambda_p': 1.0082846643230274}
    model1 = CancerModel(lambda_p=pred['lambda_p'], delta_qp=pred["delta_qp"], gamma_q=pred['gamma_q'], gamma_p=pred['gamma_p'], KDE=pred["KDE"], k_qpp=pred['k_qpp'], k_pq=pred['k_pq'], K=pred["K"])
    x = odeint(model1.model, X0, t)
    P1 = x[:, 0]
    Q1 = x[:, 1]
    Q_p1 = x[:, 2]
    C1 = x[:, 3]

    pred = {'K': 162.13542259620064, 'KDE': 0.050132007123680766, 'delta_qp': 19.356119021565608, 'gamma_p': 10.704338231684789, 'gamma_q': 1.6060018756305436, 'k_pq': 1.0095729842354653, 'k_qpp': 0.022382598683551486, 'lambda_p': 1.4871219842328882}
    model2 = CancerModel(lambda_p=pred['lambda_p'], delta_qp=pred["delta_qp"], gamma_q=pred['gamma_q'], gamma_p=pred['gamma_p'], KDE=pred["KDE"], k_qpp=pred['k_qpp'], k_pq=pred['k_pq'], K=pred["K"])
    x = odeint(model2.model, X0, t)
    P2 = x[:, 0]
    Q2 = x[:, 1]
    Q_p2 = x[:, 2]
    C2 = x[:, 3]

    pred = {'K': 122.10670000521083, 'KDE': 0.04845378325699074, 'delta_qp': 0.20603915599018055, 'gamma_p': 17.01815275513705, 'gamma_q': 1.5911157863518743, 'k_pq': 0.533740469604438, 'k_qpp': 0.010684277490226471, 'lambda_p': 1.0237525686140283}
    model3 = CancerModel(lambda_p=pred['lambda_p'], delta_qp=pred["delta_qp"], gamma_q=pred['gamma_q'], gamma_p=pred['gamma_p'], KDE=pred["KDE"], k_qpp=pred['k_qpp'], k_pq=pred['k_pq'], K=pred["K"])
    x = odeint(model3.model, X0, t)
    P3 = x[:, 0]
    Q3 = x[:, 1]
    Q_p3 = x[:, 2]
    C3 = x[:, 3]

    P = P1*weight1 + P2*weight2+P3*weight3
    Q = Q1 * weight1 + Q2 * weight2 + Q3 * weight3
    Q_p = Q_p1 * weight1 + Q_p2 * weight2 + Q_p3 * weight3
    C = C1 * weight1 + C2 * weight2 + C3 * weight3

    return {"data" :np.array([P+Q+Q_p,C],dtype=float)}

def distance1(x, y):
    return np.absolute(x["data"][0]- y["data"][0]).sum()
def distance2(x, y):
    return np.absolute(x["data"][1]- y["data"][1]).sum()

dist = pyabc.distance.AggregatedDistance([distance1,distance2])
prior = pyabc.Distribution(weight1=pyabc.RV("uniform", 0, 1), weight2=pyabc.RV("uniform",0, 1), weight3=pyabc.RV("uniform", 0, 1))

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