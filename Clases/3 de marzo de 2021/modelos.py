#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import random
import scipy
import scipy.cluster
from scipy import stats
from numpy import array
import math
import matplotlib.pyplot as plot


def septorch(datos, tipo, donde):
    entradas = [[col for col in fila[:-1]] for fila in datos]
    salidas = [fila[-1:] for fila in datos]
    redent = torch.tensor(entradas, dtype=tipo, device=donde, requires_grad=True)
    redsal = torch.tensor(salidas, dtype=tipo, device=donde)
    return redent, redsal


class Datos(data.Dataset):  # data.Dataset

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return (x, y)


# Cargamos los datos, todos
fichdatos = open('casas.trn', 'r')
datos = [[float(cada) for cada in linea.strip().split()] for linea in fichdatos.readlines()]
numentradas = len(datos[0]) - 1
datot = stats.zscore(array(datos, dtype=np.float32))
dtype = torch.float
device = torch.device("cpu")
te, ts = septorch(datot, dtype, device)
total = Datos(te, ts)
tamsubmues = 10
npruebas = 100
# selec = data.RandomSampler(total, num_samples=tamsubmues, replacement=True)
# muestras = data.DataLoader(total, batch_size=tamsubmues, sampler=selec)
muestras=data.DataLoader(total,batch_size=len(datot)//10, shuffle=True) #validaciÃ³n cruzada
rmse = np.zeros(npruebas)  # npruebas: Â¿CuÃ¡ntas extracciones?
for p, remues in zip(range(npruebas), muestras):
    ent_sub, sal_sub = remues
    # AquÃ­ ajustas la red. Supongamos que se llama red
    ocultos = 5
    red = nn.Sequential(
        # Transforma1D(numentradas, ocultos),
        # nn.Linear(numentradas, 1)
        nn.Linear(numentradas, ocultos),
        nn.Tanh(),  # Para un modelo lineal, suprimirÃ­as esto
        nn.Linear(ocultos, 1),
    )

    # .cuda(device)
    # definir error a optimizar
    error = nn.MSELoss()
    # definir algoritmo de ajuste
    ajuste = torch.optim.SGD(red.parameters(), lr=0.0005)


    def evalua():
        ajuste.zero_grad()
        s = red(ent_sub)
        exx = error(s, sal_sub)
        exx.backward()
        return exx


    print("Iteración", "Error de ajuste", "Error de validación")
    for it in range(100):  # Calcula salidas
        ent_sub.requires_grad_()
        ajuste.zero_grad()
        s = red(ent_sub)
        exx = error(s, sal_sub)
        exx.backward()
        print(it, math.sqrt(exx.item()))

        ajuste.step()

    # Prueba:
    with torch.no_grad():
        salpru = red(te)
        ep = error(salpru, ts)
        rmse[p] = math.sqrt(ep.item())
plot.hist(rmse)  # Si te llevas bien con matplotlib
plot.show()