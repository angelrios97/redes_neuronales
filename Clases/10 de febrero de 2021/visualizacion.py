import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

plot.close("all")
fichero = open("casas.trn", "r")
lineas = [_.strip().split() for _ in fichero.readlines()]
lineas = [[float(_) for _ in fila] for fila in lineas]
datos = pd.DataFrame(lineas)
print(datos)
datos2=np.array(lineas)

# for _ in range(1,17):
#     plt.figure()
#     plt.subplot(16,1,_)
#     plt.scatter(datos[_], range(len(datos[_])))
# plt.show()

def grafini(numfils, numcols):
    # Para crear un conjunto de grÃ¡ficos
    return plot.subplots(numfils, numcols, figsize=(20, 15))


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
        grafindiv(tablagraf, var1, var2, datos2[:, var1], datos2[:, var2])
# grafconc(conj, "Vars2 respecto a vars1")
plot.show()