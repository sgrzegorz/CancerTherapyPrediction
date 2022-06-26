import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data.klusek.EP1.config import threatment_start,threatment_end,threatment2_start

df = pd.read_csv("data/klusek/EP1/stats0.csv")

t_measured = np.linspace(threatment_start,threatment2_start,num=100)

# odkomentuj zeby miec zmienne samplowanie:
# t_measured = np.linspace(threatment_start,threatment_end,num=4)
# t_measured = np.concatenate((t_measured,np.linspace(threatment_end,threatment2_start,num=8)))

t_measured = np.around(t_measured)
P_measured = df.loc[t_measured,'prolif_cells'].to_list()

n = len(P_measured)
P_measured = np.array(P_measured) + (np.random.rand(n)*10000-5000)


plt.plot(list(df.iteration), list(df.prolif_cells))
plt.scatter(t_measured,P_measured,color='red')
plt.show()

print(list(t_measured))

print(list(P_measured))