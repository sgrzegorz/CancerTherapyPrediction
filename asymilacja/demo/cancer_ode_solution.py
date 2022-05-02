import numpy as np


# function that returns dy/dt
from matplotlib import pyplot as plt
from scipy.integrate import odeint

KDE = 0.3


def model(y,t):
    dydt = -KDE * y
    return dydt

# initial condition
C0 = 5


t = np.linspace(0,20)

print(t)


# solve ODE
C = odeint(model,C0,t)

# plot results
plt.plot(t,C)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title('ode')
plt.show()
from data.klusek.patient4.config import threatment_start, threatment_end, threatment2_start

t = np.linspace(0,20)

print(t)

def model1(t):
    return np.exp(-KDE*t) * C0


C1 = [model1(i) for i in t]
plt.plot(t,C1)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title('solution')
plt.show()