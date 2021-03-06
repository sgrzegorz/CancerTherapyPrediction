from scipy.signal import argrelextrema
from asymilacja.utils.datasets import klusek_dataset
import numpy as np



def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == "__main__":
    df = klusek_dataset("data/klusek/patient4/2dawki.csv")
    maximums = argrelextrema(df.prolif_cells.to_numpy(), np.greater)[0]

    k = df.dead_cells[maximums[0]] / df.prolif_cells[maximums[0]]
    df.prolif_cells = NormalizeData(list(df.prolif_cells))*50
    df.dead_cells = NormalizeData(list(df.dead_cells))*50*k


    t = [df.t[i] for i in maximums]
    # plt.plot(df.t,df.prolif_cells, label = "prolif")
    # plt.plot(df.t,df.dead_cells, label = "prolif")
    # plt.show()

    from asymilacja.utils.curement import curement_function

    df['curement'] = [curement_function(x,58.33333,169.16667) for x in df.t]
    df1 = df.copy()
    df = df[(df.t > t[0])  &  (df.t < t[1])]

    df.to_csv("asymilacja/trening/okres.csv")