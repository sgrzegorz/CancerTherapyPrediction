
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from asymilacja.model.datasets import klusek_dataset
import numpy as np


df = klusek_dataset("data/klusek/2dawki.txt")

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


maximums = argrelextrema(df.prolif_cells.to_numpy(), np.greater)[0]

k = df.dead_cells[maximums[0]] / df.prolif_cells[maximums[0]]
df.prolif_cells = NormalizeData(list(df.prolif_cells))*50
df.dead_cells = NormalizeData(list(df.dead_cells))*50*k


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

df.to_csv("asymilacja/trening/okres.csv")