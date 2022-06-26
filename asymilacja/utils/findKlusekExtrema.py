import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def plot_extrema(df,N=None): # N szerokość okna
    x = df.prolif_cells.to_numpy()
    if N is not None:
        x = np.convolve(x, np.ones(N)/N, mode='valid')
    maximums = argrelextrema(x, np.greater)[0]
    maximums = np.concatenate((maximums,argrelextrema(x, np.less)[0]))
    maximums = np.unique(maximums)

    i =maximums
    maximums = df.loc[maximums,'iteration'].to_numpy()
    maximums = np.sort(maximums)

    print(maximums)
    plt.plot(df.iteration, df.prolif_cells)
    plt.scatter(maximums,df.loc[i,'prolif_cells'])
    plt.show()


df = pd.read_csv("data/klusek/EP3/stats0.csv")
# df = pd.read_csv("data/klusek/patient4/2dawki.csv")
# df = pd.read_csv("data/klusek/patient3/stats_wszystkie_iteracje.csv")

plot_extrema(df,N=50)