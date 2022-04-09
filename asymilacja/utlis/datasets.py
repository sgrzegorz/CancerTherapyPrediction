import math

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

pd.options.mode.chained_assignment = None


def ribba_dataset(path): #'/home/x/doc/dev/master/CancerTherapyPrediction/data/fig4.csv'
    def to_volume(mtd):
        return math.pi * mtd**3/6

    def preprocess(mtd):
        val = to_volume(mtd)
        return val/1500

    df4 = pd.read_csv(path,names = ['t','mtd','id'])
    patient = df4[df4.id ==2]
    patient.mtd = patient['mtd'].map(lambda x: preprocess(x))

    return patient

def klusek_dataset(path): # 2.5. Computational model -> podane że 1 itaracja do 6min, w rezultacie 20k iteracji to około 80 dni
    header = ['iteration', 'alive_cells', 'silent_cells', 'dead_cells', 'number_of_artery_giving_oxygen', 'volume']
    df = pd.read_csv(path, sep=' ', names=header, skiprows=1)
    df['prolif_cells'] = df['alive_cells'] + df['silent_cells']
    df['t'] = df['iteration'] * 6 /60. /24.0 # convert iterations to days
    return df


# def partition_klusek(df):
#     maximums = argrelextrema(df.prolif_cells.to_numpy(), np.greater)
#     x = [df.t[i] for i in maximums]
#     y = [df.prolif_cells[i] for i in maximums]

# df = klusek_dataset("data/klusek/stats0.txt")