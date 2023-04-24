#!/usr/bin/python3

"""Equilibrio ponderado en el conjunto de ajuste. Repetición y media."""
import copy

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
mindatclase = 10
muesajuste = 0.8
muesval = 0.5  # muespru = 1 - muesval
dtype = torch.long
device = torch.device('cuda:0')
iter = 10
error_serie = np.zeros(iter)
error_serie2 = np.zeros(iter)

def inicial(capa): #Inicializa cada capa
    try:
        capa.reset_parameters()
    except AttributeError:
        pass
for i in range(iter):
    if 'red' in dir():
        red = copy.deepcopy(red).cuda(device)
        red.apply(inicial)
    datos = zscore(array(matriz[:, 1:], dtype=np.float32))
    datos = np.hstack([np.atleast_2d(clase).T, datos])  # Datos normalizados junto a la clase.
    peso_clase = list(pd.crosstab(clase, 'freq')['freq'])  # Frecuencias de la clase en la muestra
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
    npruebas = 1
    sampler = data.WeightedRandomSampler(peso_instancia, num_samples=tamsubmues * npruebas, replacement=True)
    muestras = data.DataLoader(total, batch_size=tamsubmues, sampler=sampler)
    # sampler = data.RandomSampler(total, num_samples=tamsubmues * npruebas, replacement=True)
    # muestras = data.DataLoader(total, batch_size=tamsubmues, sampler=sampler)
    numvarent = len(te[0])
    tea, tsa = list(muestras)[0]

    ocultos = int(sqrt(numvarent * nnclases))
    red = nn.Sequential(
        nn.Linear(numvarent, ocultos),
        nn.Hardtanh(),
        nn.Linear(ocultos, nnclases),
    ).cuda(device)  # 1.7

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
    ajuste = torch.optim.RMSprop(red.parameters(), lr=0.00001, alpha=0.9, eps=1e-06); nvalfalmax = 3  # 0.00001  3

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
        print(i, 1, it, ea.item(), evprevio, errorclas(salval, tsv))
        ajuste.step(evalua)
# Prueba:
    with torch.no_grad():
        salpru = red(tep.to(torch.float))
        ep = error(salpru, tsp)
        error_serie[i] = ep.item()


    salida = red(torch.tensor(datos, device=device, dtype=torch.float)[:, 1:])
    datos = torch.tensor(np.hstack([datos, salida.cpu().detach().numpy()]), device=device, dtype=dtype)

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
    npruebas = 1
    sampler = data.WeightedRandomSampler(peso_instancia, num_samples=tamsubmues * npruebas, replacement=True)
    muestras = data.DataLoader(total, batch_size=tamsubmues, sampler=sampler)
    # sampler = data.RandomSampler(total, num_samples=tamsubmues * npruebas, replacement=True)
    # muestras = data.DataLoader(total, batch_size=tamsubmues, sampler=sampler)
    numvarent = len(te[0])
    errores = np.zeros(npruebas)
    tea, tsa = list(muestras)[0]

    ocultos = int(sqrt(numvarent * nnclases))
    red = nn.Sequential(
        nn.Linear(numvarent, ocultos),
        nn.Hardtanh(),
        nn.Linear(ocultos, nnclases),
    ).cuda(device)

    ajuste = torch.optim.RMSprop(red.parameters(), lr=0.0001, alpha=0.9); nvalfalmax = 3

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
        print(i, 2, it, ea.item(), evprevio, errorclas(salval, tsv))
        ajuste.step(evalua)
    # Prueba:
    with torch.no_grad():
        salpru = red(tep.to(torch.float))
        ep = error(salpru, tsp)
        error_serie2[i] = ep.item()


print("Errores red", error_serie, "Media", np.mean(error_serie))  # 1.7
print("Errores redes en serie", error_serie2, "Media", np.mean(error_serie2))  # 1.4
