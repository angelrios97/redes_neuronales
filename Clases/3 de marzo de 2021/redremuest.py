#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
PerceptrÃ³n para estimar precio de viviendas.
"""

import math
import torch
import torch.nn as nn
import random
import scipy
import scipy.cluster
from scipy import stats
import numpy as np
from numpy import array


def septorch(datos, tipo, donde):
    entradas = datos[:, :-1]
    salidas = datos[:, -1:]
    redent = torch.tensor(entradas, dtype=tipo, device=donde)
    redsal = torch.tensor(salidas, dtype=tipo, device=donde)
    return redent, redsal


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")

# Cargamos los datos, todos
fichdatos = open('casas.trn', 'r')
datos = [[float(cada) for cada in linea.strip().split()] for linea in fichdatos.readlines()]
numentradas = len(datos[0]) - 1

# Separamos ajuste y prueba
porazar = 0.4
numuestras = len(datos)
muesajuste=0.8
muesval=0.1
numestim = 8
limaj = np.zeros(numestim)
difer = np.zeros(numestim)
for v in range(numestim):
    # Desordena conjunto
    random.shuffle(datos)
    # Separa una parte para escoger por azar
    limazar = int(porazar * numuestras)
    datazar = datos[:limazar]
    datgrup = datos[limazar:]

    # Separa un primer lote de ajuste y prueba por azar
    limajaz = int(limazar * muesajuste)
    limvalaz = int(limazar * (muesajuste + muesval))
    datajaz = datazar[:limajaz]
    datvalaz = datazar[limajaz:limvalaz]
    datpruaz = datazar[limvalaz:]
    # Separa un segundo lote de ajuste y prueba por agrupamiento
    datkm = array(datgrup, dtype=np.float32)
    limgrupaj = len(datgrup)
    numajgrup = int(limgrupaj * muesajuste)
    centros, grupos = scipy.cluster.vq.kmeans2(datkm, numajgrup, minit='points')
    orgpuntos = scipy.spatial.KDTree(datkm)
    dist, ind = orgpuntos.query(centros)
    datajgrup = datkm[ind]
    indvalpru = np.setdiff1d(range(limgrupaj), ind)
    datvalprugrup = datkm[indvalpru]
    numprugrup = int(limgrupaj * (1 - muesval - muesajuste))
    centros, grupos = scipy.cluster.vq.kmeans2(datvalprugrup, numprugrup, minit='points')
    orgpuntos = scipy.spatial.KDTree(datvalprugrup)
    dist, ind = orgpuntos.query(centros)
    datprugrup = datvalprugrup[ind]
    indpru = np.setdiff1d(range(len(datvalprugrup)), ind)
    datvalgrup = datvalprugrup[indpru]

    dataj = np.vstack((array(datajaz, dtype=np.float32), datajgrup))
    datval = np.vstack((array(datvalaz, dtype=np.float32), datvalgrup))
    datpru = np.vstack((array(datpruaz, dtype=np.float32), datprugrup))

    # Pasarlo a tensores torch
    tea, tsa = septorch(dataj, dtype, device)
    tev, tsv = septorch(datval, dtype, device)
    tep, tsp = septorch(datpru, dtype, device)

    # definir red
    ocultos = 200
    red = nn.Sequential(
        nn.Linear(numentradas, ocultos),
        nn.Tanh(),
        nn.Linear(ocultos, 1),
    )
    # ).cuda(device)
    # definir error a optimizar
    error = nn.MSELoss()

    # definir algoritmo de ajuste
    ajuste=torch.optim.LBFGS(red.parameters(),lr=0.001,tolerance_grad=1e-03, tolerance_change=1e-03, max_iter=100,history_size=10) #Más parámetros en cuasinewton


    nvalfal = 0


    def evalua():
        ajuste.zero_grad()
        s = red(tea)
        e = error(s, tsa)
        e.backward()
        return e


    for it in range(1500):
        ea = evalua()
        salval = red(tev)
        ev = error(salval, tsv)
        if 'evprevio' in dir():
            if evprevio < ev.item():
                nvalfal = nvalfal + 1
            else:
                nvalfal = 0
        if nvalfal > 2:
            break
        evprevio = ev.item()

        ajuste.step(evalua)

    limaj[v] = ea.item()
    # Prueba
    salpru = red(tep)
    ep = error(salpru, tsp)
    print(it, math.sqrt(ep.item()))  # Si no usas el cuadrÃ¡tico quita la raÃ­z
    difer[v] = ep.item() - ea.item()   # DIFERENCIA ENTRE EL AJUSTE Y LA PRUEBA


tea, tsa = septorch(array(datos, dtype=np.float32), dtype, device)
# definir red
ocultos = 10
red = nn.Sequential(
    nn.Linear(numentradas, ocultos),
    nn.Tanh(),
    nn.Linear(ocultos, 1),
)
# ).cuda(device)
# definir error a optimizar
error = nn.MSELoss()

# definir algoritmo de ajuste
ajuste=torch.optim.LBFGS(red.parameters(),lr=0.001,tolerance_grad=1e-03, tolerance_change=1e-03, max_iter=100,history_size=10) #Más parámetros en cuasinewton



def evalua():
    ajuste.zero_grad()
    s = red(tea)
    e = error(s, tsa)
    e.backward()
    return e


tope = np.median(limaj)  # o mean o ....
for it in range(6000):
    ea = evalua()
    if ea.item() < tope in dir():
        break
    print(it, math.sqrt(ea.item()))  # Si no usas el cuadrÃ¡tico quita la raÃ­z

    ajuste.step(evalua)

print(math.sqrt(ea.item() + np.median(difer)))  # Si no usas el cuadrÃ¡tico quita la raÃ­z
