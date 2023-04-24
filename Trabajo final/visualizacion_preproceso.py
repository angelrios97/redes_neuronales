##
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plot
import scipy.stats as stat
from statsmodels.stats.multitest import multipletests as mtest
from functools import partial

## Cargar los datos.
datos = sio.loadmat('ajuste.mat')['muesaj']

## Primeras instancias en la muestra

for _ in range(4):
    plot.subplot(2, 2, _ + 1)
    plot.plot(range(len(datos[_])), datos[_])
    plot.title("Instancia " + str(_ + 1))
plot.suptitle("Primeras instancias de la muestra")
#plot.plot(range(len(datos[0])), datos[0])
plot.show()
plot.pause(5)

## Frecuencia del número de datos recogidos por persona.
freq = pd.crosstab(datos[:, 0], columns='freq')
print(freq)
plot.title('Número de instancias por persona', fontsize=16)
plot.bar(freq['freq'].keys(),freq['freq'])
plot.xlim(-1, 38)
plot.xlabel("Individuo (núm)")
plot.yticks(np.arange(0,95, step=10))
plot.ylabel("Frecuencia")
# plot.show()
plot.pause(5)  # Segundos
#plot.close()

# Observamos que tenemos 37 individuos con muy distinto número de muestras para cada uno. Se trata de un problema
# de minería de datos supervisada para clasificación con 37 clases no equilibradas. Escogeremos el error entrópico
# medio porque se usa frecuentemente para clasificación y tenemos referencias de los niveles obtenidos.

# Ponderar los errores podría no ser suficiente porque hay clases con un número muy bajo de individuos en la muestra.
# Si el algoritmo no los escoge para ajustar, entonces no podrá predecirlos. Si el algoritmo los escoge todos para
# ajustar, entonces no se le pedirá predecirlos.

## Visualización de la trayectoria del ojo para un par de muestras de individuos.

indiv = datos[datos[:, 0] == 2]
k = 0
for _ in range(4):
    k += 1
    plot.subplot(4, 2, k)
    plot.plot(indiv[_, 1:2049], indiv[_, 4097:6145], color='g')
    plot.title('Instancia ' + str(_ + 1) + ' ojo izquierdo', fontsize=9)
    k += 1
    plot.subplot(4, 2, k)
    plot.plot(indiv[_, 2049:4097], indiv[_, 6145:], color='r')
    plot.title('Instancia ' + str(_ + 1) + ' ojo derecho', fontsize=9)
plot.suptitle('Recorrido de los ojos en el individuo 2', fontsize=16)
plot.subplots_adjust(hspace=0.4)
plot.show()

##
# indiv = datos[datos[:, 0] == 2]
# k = 0
# for _ in range(4):
#     k += 1
#     plot.subplot(4, 2, k)
#     plot.plot(indiv[_, 1:2049], indiv[_, 2049:4097], color='g')
#     plot.title('Instancia ' + str(_ + 1) + ' ojo izquierdo', fontsize=9)
#     k += 1
#     plot.subplot(4, 2, k)
#     plot.plot(indiv[_, 4097:6145], indiv[_, 6145:], color='r')
#     plot.title('Instancia ' + str(_ + 1) + ' ojo derecho', fontsize=9)
# plot.suptitle('Recorrido de los ojos en el individuo 2', fontsize=16)
# plot.subplots_adjust(hspace=0.4)
# plot.show()
# plot.pause(8)
# plot.close()



## Comparamos las medias muestrales de cada una de las 8192 variables.

mediaslx = np.array([])
mediasrx = np.array([])
mediasly = np.array([])
mediasry = np.array([])
for _ in range(1, 2049):
    mediaslx = np.append(mediaslx, np.mean(datos[:, _]))
    mediasrx = np.append(mediasrx, np.mean(datos[:, _ + 2048]))
    mediasly = np.append(mediasly, np.mean(datos[:, _ + 2048 * 2]))
    mediasry = np.append(mediasry, np.mean(datos[:, _ + 2048 * 3]))
print(mediaslx-mediasrx)  # Las medias muestrales no son iguales.
print(mediasly-mediasry)  # Las medias muestrales no son iguales.

fig, ax = plot.subplots(1, 4, sharex=True, sharey=True)
fig.suptitle('Medias de abscisas y ordenadas en cada ojo')
plotbar = partial(plot.bar, x=range(2048))
ax[0].bar(range(2048), mediaslx)
ax[0].set_title('Medias de abscisas ojo izquierdo')
ax[1].bar(range(2048), mediasly)
ax[1].set_title('Medias de ordenadas ojo izquierdo')
ax[2].bar(range(2048), mediasrx)
ax[2].set_title('Medias de abscisas ojo derecho')
ax[3].bar(range(2048), mediasry)
ax[3].set_title('Medias de ordenadas ojo derecho')
plot.pause(5)


## Varianzas de las 8192 variables

varianzalx = np.array([])
varianzarx = np.array([])
varianzaly = np.array([])
varianzary = np.array([])
for _ in range(1, 2049):
    varianzalx = np.append(varianzalx, np.var(datos[:, _]))
    varianzarx = np.append(varianzarx, np.var(datos[:, _ + 2048]))
    varianzaly = np.append(varianzaly, np.var(datos[:, _ + 2048 * 2]))
    varianzary = np.append(varianzary, np.var(datos[:, _ + 2048 * 3]))

fig, ax = plot.subplots(1, 4, sharex=True, sharey=True)
fig.suptitle('Varianzas de abscisas y ordenadas en cada ojo')
plotbar = partial(plot.bar, x=range(2048))
ax[0].bar(range(2048), varianzalx)
ax[0].set_title('Varianzas de abscisas ojo izquierdo')
ax[1].bar(range(2048), varianzaly)
ax[1].set_title('Varianzas de ordenadas ojo izquierdo')
ax[2].bar(range(2048), varianzarx)
ax[2].set_title('Varianzas de abscisas ojo derecho')
ax[3].bar(range(2048), varianzary)
ax[3].set_title('Varianzas de ordenadas ojo derecho')
plot.pause(5) # Observamos como esperábamos que las abscisas del ojo derecho tienen varianza mucho mayor.


## Test de normalidad de cada una de las 8192 variables.

pvalores = np.asarray(list(map(lambda x: stat.shapiro(x)[1], np.transpose(datos[:,1:]))))
pvalores_cor = mtest(pvalores)
print(pd.value_counts(pvalores_cor[0]))  # Rechazaríamos la hipótesis de normalidad 7726 veces al nivel 0.05 FWER


## Test no paramétrico de igualdad de distribución entre el ojo izquierdo y el derecho para abscisas y ordenadas.

pvalorx = np.asarray(list(map(lambda x, y: stat.wilcoxon(x, y)[1],
                               np.transpose(datos[:, 1:2049]),
                               np.transpose(datos[:, 2049:4097]))))
pvalory = np.asarray(list(map(lambda x, y: stat.wilcoxon(x, y)[1],
                               np.transpose(datos[:, 4097:6145]),
                               np.transpose(datos[:, 6145:]))))
pvalorx_cor = mtest(pvalorx)
pvalory_cor = mtest(pvalory)
print(pd.value_counts(pvalorx_cor[0]))  # Rechazaríamos la igualdad de distribución 1928 veces al nivel 0.05 FWER
print(pd.value_counts(pvalory_cor[0]))  # Rechazaríamos la igualdad de distribución 1387 veces al nivel 0.05 FWER

# En general, los datos del ojo izquierdo no se distribuyen igual que los del derecho.

##

