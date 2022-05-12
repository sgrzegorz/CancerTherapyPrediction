from scipy.signal import argrelextrema
from asymilacja.utils.datasets import klusek_dataset
import numpy as np


df = klusek_dataset("data/klusek/patient4/2dawki.csv")


maximums = argrelextrema(df.prolif_cells.to_numpy(), np.greater)[0]



t = [df.t[i] for i in maximums]
# plt.plot(df.t,df.prolif_cells, label = "prolif")
# plt.plot(df.t,df.dead_cells, label = "prolif")
# plt.show()

def curement_function(x,x1,x2):
    y1 = 1
    y2 = 0
    if(x1 == x2):
        raise RuntimeError("x1 == x2")
    if x < x1 or x > x2:
        return 0

    return (y2 - y1) / (x2-x1)* x + (x2*y1 - x1*y2)/(x2-x1)

df['curement'] = [curement_function(x,58.33333,169.16667) for x in df.t]
df1 = df.copy()
df = df[(df.t > t[0])  &  (df.t < t[1])]
# df = df[(df.t <= t[0])]


df.to_csv("data/klusek/patient4/2dawki_p2.csv")

# df = pd.read_csv("asymilacja/trening/okres.csv")
# df1 = pd.DataFrame()
# df1['t'] = list(df['t'])
# df1['dead_cells'] = list(df['dead_cells'])
# df1['curement'] = list(df['curement'])
# df1['prolif_cells'] = list(df['prolif_cells'])
# df1.to_csv("asymilacja/trening/okres1.csv")