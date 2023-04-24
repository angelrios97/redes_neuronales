#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Perceptrón para estimar precio de viviendas.
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

def septorch(datos,tipo,donde):
  entradas=datos[:,:-1]
  salidas=datos[:,-1:]
  redent=torch.tensor(entradas,dtype=tipo,device=donde)
  redsal=torch.tensor(salidas,dtype=tipo,device=donde)
  return redent,redsal

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")

#Cargamos los datos, todos
fichdatos=open('casas.trn','r')
datos= [[float(cada) for cada in linea.strip().split()] for linea in fichdatos.readlines()]
numentradas=len(datos[0])-1

#Separamos ajuste y prueba
porazar=0.4
numuestras=len(datos)
muesajuste=0.8
muesval=0.1
#Desordena conjunto
random.shuffle(datos)
#Separa una parte para escoger por azar
limazar=int(porazar*numuestras)
datazar=datos[:limazar]
datgrup=datos[limazar:]

#Separa un primer lote de ajuste y prueba por azar
limajaz=int(limazar*muesajuste)
limvalaz=int(limazar*(muesajuste+muesval))
datajaz=datazar[:limajaz]
datvalaz=datazar[limajaz:limvalaz]
datpruaz=datazar[limvalaz:]
#Separa un segundo lote de ajuste y prueba por agrupamiento
datkm=array(datgrup,dtype=np.float32)
limgrupaj=len(datgrup)
numajgrup=int(limgrupaj*muesajuste)
centros,grupos=scipy.cluster.vq.kmeans2(datkm,numajgrup,minit='points')
orgpuntos=scipy.spatial.KDTree(datkm)
dist,ind=orgpuntos.query(centros)
datajgrup=datkm[ind]
indvalpru=np.setdiff1d(range(limgrupaj),ind)
datvalprugrup=datkm[indvalpru]
numprugrup=int(limgrupaj*(1-muesval-muesajuste))
centros,grupos=scipy.cluster.vq.kmeans2(datvalprugrup,numprugrup,minit='points')
orgpuntos=scipy.spatial.KDTree(datvalprugrup)
dist,ind=orgpuntos.query(centros)
datprugrup=datvalprugrup[ind]
indpru=np.setdiff1d(range(len(datvalprugrup)),ind)
datvalgrup=datvalprugrup[indpru]

dataj=np.vstack((array(datajaz,dtype=np.float32),datajgrup))
datval=np.vstack((array(datvalaz,dtype=np.float32),datvalgrup))
datpru=np.vstack((array(datpruaz,dtype=np.float32),datprugrup))

#Pasarlo a tensores torch
tea,tsa=septorch(dataj,dtype,device)
tev,tsv=septorch(datval,dtype,device)
tep,tsp=septorch(datpru,dtype,device)


a = 10
b = 20
c = 5
#definir red
red=nn.Sequential(
    nn.Linear(numentradas,a),
    nn.Tanh(),
    nn.Dropout(p=0.05),
    nn.Linear(a, b),
    nn.Tanh(),
    nn.Dropout(p=0.05),
    nn.Linear(b,c),
    nn.Tanh(),
    nn.Dropout(p=0.05),
    nn.Linear(c, 1),
)
#).cuda(device)
#definir error a optimizar
error = nn.MSELoss()

#definir algoritmo de ajuste
ajuste= torch.optim.SGD(red.parameters(), lr=0.0005)
# ajuste=torch.optim.Rprop(red.parameters(),lr=0.0005)

nvalfal=0
def evalua():
        ajuste.zero_grad()
        s = red(tea)
        e = error(s, tsa)
        e.backward()
        return e
for it in range(2500):
  ea=evalua()
  salval = red(tev)
  ev=error(salval,tsv)
  if 'evprevio' in dir():
    if evprevio<ev.item():
      nvalfal=nvalfal+1
    else:
      nvalfal=0
  if nvalfal>1800:
    break
  evprevio=ev.item()
  print(it, math.sqrt(ea.item()),math.sqrt(evprevio)) #Si no usas el cuadrático quita la raíz

  ajuste.step(evalua)

#Prueba
salpru=red(tep)
ep=error(salpru,tsp)
print(math.sqrt(ep.item())) #Si no usas el cuadrático quita la raíz
