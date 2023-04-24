#!/usr/bin/python3

"""Equilibrio ponderado en el conjunto de ajuste. Repetición y media."""

import numpy as np
from numpy import array
import pandas as pd
import scipy.io as sio
from scipy.stats import zscore
from random import shuffle
import torch
import torch.nn as nn
import torch.types
import torch.utils.data as data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.model_selection import train_test_split
from math import sqrt


def septorch(data, tipo, donde):  # Crea tensores torch de entrada y salida
    entradas = data[:, 1:]
    salidas = data[:, 0]
    redent = torch.tensor(entradas, dtype=tipo, device=donde)
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
matriz = sio.loadmat('ajuste.mat')['muesaj']
# matriz = np.concatenate((matriz[:, :2049], matriz[:, 4097:6145]), axis=1)  # SOLO OJO IZQUIERDO
clase = matriz[:, 0] - 1
datos = zscore(array(matriz[:, 1:], dtype=np.float32))
datos = np.hstack([np.atleast_2d(clase).T, datos])  # Datos normalizados junto a la clase.
peso_clase = list(pd.crosstab(clase, 'freq')['freq'])  # Frecuencias de la clase en la muestra
mindatclase = 5
muesajuste = 0.8
muesval = 0.5  # muespru = 1 - muesval
dtype = torch.long
device = torch.device('cuda:0')
if mindatclase >= 2:
    escogidas = [float(_) for _ in range(len(peso_clase)) if peso_clase[_] >= mindatclase]
    nnclases = len(escogidas) + 1
    nuevas_clases = {k: v for k, v in zip(escogidas, range(nnclases - 1))}
    nuevas_clases.update({float(k): (nnclases - 1) for k in range(len(peso_clase)) if k not in escogidas})
    for _ in range(len(datos)):
        datos[_, 0] = nuevas_clases[datos[_, 0]]
    nueva_clase = datos[:, 0]
    peso_nueva_clase = pd.crosstab(nueva_clase, 'freqn')['freqn']
    peso_instancia_datos = [1/peso_nueva_clase[int(datos[_][0])] for _ in range(len(datos))]
    dataj, datvalpru = train_test_split(datos, test_size= 1 - muesajuste, stratify=peso_instancia_datos)
    peso_instancia_datvalpru = [1/peso_nueva_clase[int(datvalpru[_][0])] for _ in range(len(datvalpru))]
    datval, datpru = train_test_split(datvalpru, test_size=muesval, stratify=peso_instancia_datvalpru)
    peso_instancia = [1 / peso_nueva_clase[int(dataj[_][0])] for _ in range(len(dataj))]
else:
    nnclases = 37
    numuestras = len(datos)
    shuffle(datos)
    limajaz = int(muesajuste * numuestras)
    limval = int(muesval * (numuestras - limajaz))
    dataj = datos[:limajaz]
    peso_instancia = [1 / peso_clase[int(dataj[_][0])] for _ in range(len(dataj))]
    datval = datos[limajaz:][:limval]
    datpru = datos[limajaz:][limval:]
te, ts = septorch(dataj, dtype, device)
tev, tsv = septorch(datval, dtype, device)
tep, tsp = septorch(datpru, dtype, device)
total = Datos(te, ts)
tamsubmues = len(te)
npruebas = 3
sampler = data.WeightedRandomSampler(peso_instancia, num_samples=tamsubmues * npruebas, replacement=True)
muestras = data.DataLoader(total, batch_size=tamsubmues, sampler=sampler)
# sampler = data.RandomSampler(total, num_samples=tamsubmues * npruebas, replacement=True)
# muestras = data.DataLoader(total, batch_size=tamsubmues, sampler=sampler)
numvarent = len(te[0])
errorred = np.zeros(npruebas)
errorlineal = np.zeros(npruebas)
for p, remues in zip(range(npruebas), muestras):
    tea, tsa = remues

    # a = numvarent * 2
    # red = nn.Sequential(
    #     nn.Linear(numvarent, a),
    #     nn.Hardtanh(),
    #     nn.Dropout(0.2),
    #     nn.Linear(a, numvarent // 2),
    #     nn.Hardtanh(),
    #     nn.Dropout(0.1),
    #     nn.Linear(numvarent // 2, numvarent // 4),
    #     nn.Hardtanh(),
    #     nn.Linear(numvarent // 4, nnclases)
    # ).cuda(device)

    ocultos = int(sqrt(numvarent * nnclases))
    # ocultos = int(2/3 * numvarent + nnclases)
    red = nn.Sequential(
        nn.Linear(numvarent, ocultos),
        nn.Hardtanh(),
        nn.Linear(ocultos, nnclases),
    ).cuda(device)


    # ocultos = int(sqrt(4097 * nnclases))
    # ocultos = 3073
    # act = nn.Hardtanh
    # red = nn.Sequential(
    #     nn.Linear(numvarent, ocultos),
    #     nn.Hardtanh(),
    #     nn.Dropout(0.1),
    #     nn.LeakyReLU(),
    #     nn.Dropout(0.1),
    #     nn.Hardtanh(),
    #     nn.LeakyReLU(),
    #     nn.Linear(ocultos, nnclases)
    # ).cuda(device)

    # act = nn.Hardtanh
    # red = nn.Sequential(
    #     nn.Linear(numvarent, numvarent * 2),
    #     act(),
    #     nn.Linear(numvarent * 2, 2770),
    #     act(),
    #     nn.Linear(2770, nnclases)
    # ).cuda(device)

    ##  Error a optimizar
    # pesos_error = torch.tensor([1/_ for _ in peso_nueva_clase], dtype=torch.float32, device=device)
    # error = nn.CrossEntropyLoss(weight=pesos_error)
    error = nn.CrossEntropyLoss()

    ## Algoritmo de ajuste
    # ajuste = torch.optim.Adam(red.parameters(), lr=0.00001)
    # ajuste = torch.optim.Adadelta(red.parameters(), lr=0.1); nvalfalmax = 3
    # ajuste = torch.optim.Adagrad(red.parameters(), lr=0.0001); nvalfalmax = 3
    # ajuste = torch.optim.ASGD(red.parameters(), lr=0.1)  # NO
    # ajuste = torch.optim.SGD(red.parameters(), lr=0.0001, momentum=0.8);  nvalfalmax = 3# NO
    ajuste = torch.optim.RMSprop(red.parameters(), lr=0.00001, alpha=0.9); nvalfalmax = 3  # 0.00001  3
    nvalfalmax = 3  # 0.00001  3
    # ajuste = torch.optim.LBFGS(red.parameters(), lr=0.05) # NO

    def clases(matrix):  # Definimos la clase como la mayor
        if len(matrix.shape) == 1:
            return matrix
        valor, clase = torch.max(matrix, 1)
        return clase


    def matconf(matsal, matreal):  # Matriz de confusión
        clasesal = clases(matsal)
        clasereal = clases(matreal)
        numclases = len(matsal[0])
        numcasos = len(clasereal)
        relacion = [[0 for col in range(numclases)] for fil in range(numclases)]
        for caso in range(numcasos):
            relacion[clasereal[caso]][clasesal[caso]] = relacion[clasereal[caso]][clasesal[caso]] + 1
        return relacion


    def errorclas(matsal, matreal):  # Error de clasificación
        clasesal = clases(matsal)
        clasereal = clases(matreal)
        err = [(clasesal[caso] != clasereal[caso]).item() for caso in range(len(clasereal))]
        return sum(err) / float(len(clasereal))


    def evalua():  # Error de validación en la red neuronal
        ajuste.zero_grad()
        s = red(tea.to(torch.float))
        e = error(s, tsa)
        e.backward()
        return e


    ## Ajuste de la red neuronal

    nvalfal = 0
    for it in range(5000):  # Más iteraciones supone más tiempo.
        ea = evalua()
        salval = red(tev.to(torch.float))
        ev = error(salval, tsv)
        if 'evprevio' in dir():
            if evprevio < ev.item():
                nvalfal = nvalfal + 1
            else:
                nvalfal = 0
        if nvalfal > nvalfalmax:  # ¿Cuántos fallos de validación seguidos?
            break
        evprevio = ev.item()
        print("R", p, it, ea.item(), evprevio, errorclas(salval, tsv), errorred)
        ajuste.step(evalua)
# Prueba:
    with torch.no_grad():
        salpru = red(tep.to(torch.float))
        ep = error(salpru, tsp)
        errorred[p] = ep.item()
errorred[npruebas - 1] = ep.item()  # Añadimos el último error
print(errorred)
print(np.mean(errorred))
print(nnclases)

for p, remues in zip(range(npruebas), muestras):
    tea, tsa = remues

    red = nn.Linear(numvarent, nnclases).cuda(device)

    ##  Error a optimizar
    # pesos_error = torch.tensor([1/_ for _ in peso_nueva_clase], dtype=torch.float32, device=device)
    # error = nn.CrossEntropyLoss(weight=pesos_error)
    error = nn.CrossEntropyLoss()

    ## Algoritmo de ajuste
    # ajuste = torch.optim.Adam(red.parameters(), lr=0.0001)
    # ajuste = torch.optim.Adadelta(red.parameters(), lr=0.1); nvalfalmax = 3
    # ajuste = torch.optim.Adagrad(red.parameters(), lr=0.0001); nvalfalmax = 3
    # ajuste = torch.optim.ASGD(red.parameters(), lr=0.01)  # NO
    # ajuste = torch.optim.SGD(red.parameters(), lr=0.01, momentum=0.8);  nvalfalmax = 3# NO
    ajuste = torch.optim.RMSprop(red.parameters(), lr=0.00001, alpha=0.9)
    nvalfalmax = 3  # 0.00001  3

    def clases(matrix):  # Definimos la clase como la mayor
        if len(matrix.shape) == 1:
            return matrix
        valor, clase = torch.max(matrix, 1)
        return clase


    def matconf(matsal, matreal):  # Matriz de confusión
        clasesal = clases(matsal)
        clasereal = clases(matreal)
        numclases = len(matsal[0])
        numcasos = len(clasereal)
        relacion = [[0 for col in range(numclases)] for fil in range(numclases)]
        for caso in range(numcasos):
            relacion[clasereal[caso]][clasesal[caso]] = relacion[clasereal[caso]][clasesal[caso]] + 1
        return relacion


    def errorclas(matsal, matreal):  # Error de clasificación
        clasesal = clases(matsal)
        clasereal = clases(matreal)
        err = [(clasesal[caso] != clasereal[caso]).item() for caso in range(len(clasereal))]
        return sum(err) / float(len(clasereal))


    def evalua():  # Error de validación en la red neuronal
        ajuste.zero_grad()
        s = red(tea.to(torch.float))
        e = error(s, tsa)
        e.backward()
        return e


    ## Ajuste de la red neuronal

    nvalfal = 0
    for it in range(5000):  # Más iteraciones supone más tiempo.
        ea = evalua()
        salval = red(tev.to(torch.float))
        ev = error(salval, tsv)
        if 'evprevio' in dir():
            if evprevio < ev.item():
                nvalfal = nvalfal + 1
            else:
                nvalfal = 0
        if nvalfal > nvalfalmax:  # ¿Cuántos fallos de validación seguidos?
            break
        evprevio = ev.item()
        print("L", p, it, ea.item(), evprevio, errorclas(salval, tsv), errorlineal)
        ajuste.step(evalua)
# Prueba:
    with torch.no_grad():
        salpru = red(tep.to(torch.float))
        ep = error(salpru, tsp)
        errorlineal[p] = ep.item()
errorlineal[npruebas - 1] = ep.item()  # Añadimos el último error
print(errorlineal)
print(np.mean(errorlineal))

print("Errores red", errorred, "Media", np.mean(errorred))
print("Errores lineal", errorlineal, "Media", np.mean(errorlineal))
plt.hist(errorred)
plt.hist(errorlineal)
plt.show()