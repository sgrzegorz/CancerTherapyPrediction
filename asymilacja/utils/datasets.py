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

def convertIterationsToDays(iterations):
    return iterations* 6 / 60. / 24.0

def preprocess_klusek_dataset(inPath,outPath):
    header = ['iteration', 'alive_cells', 'silent_cells', 'dead_cells', 'number_of_artery_giving_oxygen', 'volume','curement']
    df = pd.read_csv(inPath, sep=' ', names=header, skiprows=1,index_col=None)
    df['prolif_cells'] = df['alive_cells'] + df['silent_cells']
    df = df.drop('silent_cells', 1)
    df = df.drop('alive_cells', 1)
    # df.index = df.index-139
    df['t'] = convertIterationsToDays(df['iteration'])

    # df["meta"] = np.nan
    df.to_csv(outPath,index=False)

# def partition_klusek(df):
#     maximums = argrelextrema(df.prolif_cells.to_numpy(), np.greater)
#     x = [df.t[i] for i in maximums]
#     y = [df.prolif_cells[i] for i in maximums]

if __name__ == '__main__':
    # preprocess_klusek_dataset("data/klusek/patient4/2dawki.txt")
    # preprocess_klusek_dataset("data/klusek/patient3/stats_wszystkie_iteracje.txt","data/klusek/patient3/stats_wszystkie_iteracje.csv")
    # preprocess_klusek_dataset("data/klusek/patient202205041854/stats0.txt","data/klusek/patient202205041854/stats0.csv")

    pass