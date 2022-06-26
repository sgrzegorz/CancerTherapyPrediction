def PlotOneSerie(t,__Observations, label=None,color=None):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (10, 4)
    #
    plt.figure()
    plt.plot(t,__Observations,label = label,color = color)
    plt.xlabel('iteracja')
    plt.ylabel('Liczba komórek nowotworowych')