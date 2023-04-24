import torch
import numpy as np
from numpy import array
import matplotlib.pyplot as plot

def grafini(numfils, numcols):
    # Para crear un conjunto de grÃ¡ficos
    return plot.subplots(numfils, numcols, sharex='col', sharey='row', figsize=(20, 15))


def grafindiv(figind, f, c, x, y):
    # Gráfico x/y en la fila f columna c
    figind[f, c].scatter(x, y)
    figind[f, c].minorticks_off()
    figind[f, c].locator_params(tight=True, nbins=4)


def grafconc(rotulo):
    # Poner título general y mostrar
    plot.suptitle(rotulo)
    plot.show()

numvar1 = 13
numvar2 = 13
conj, tablagraf = grafini(numvar1, numvar2)
for var1 in range(numvar1):
    for var2 in range(numvar2):
        grafindiv(tablagraf, var1, var2, datos[:, var1], datos[:, var2])
grafconc(conj, "Vars2 respecto a vars1")
