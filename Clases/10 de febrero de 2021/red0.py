#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
PerceptrÃ³n para estimar precio de viviendas. Una capa oculta con 5 procesadores con activaciÃ³n tangente hiperbÃ³lica y error cuadrÃ¡tico.
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
import matplotlib.pyplot as plot

class Transforma1D(nn.Module):
    """
MÃ³dulo consistente en varias minirredes en paralelo, cada una trabajando con una entrada y concatenando sus salidas
    """
    def __init__(self,numentradas,numinter):
        #ramas son los bloques separados que actÃºan
        #cada bloque es a su vez una secuencia
        super().__init__()
        self.ramas = nn.ModuleList([nn.Sequential(nn.Linear(1, numinter),nn.Tanh(),nn.Linear(numinter, 1))]*numentradas)

    def forward(self, x):
        #cada rama opera independientemente sobre una entrada y luego se concatenan sus resultados
        resuls=list()
        for ent in range(len(self.ramas)):
            resuls.append(self.ramas[ent](x[:,ent:ent+1]))
        return torch.cat(resuls,1)


def septorch(datos,tipo,donde):
  entradas=[ [col for col in fila[:-1]] for fila in datos]
  salidas=[ fila[-1:] for fila in datos]
  redent=torch.tensor(entradas,dtype=tipo,device=donde,requires_grad=True)
  redsal=torch.tensor(salidas,dtype=tipo,device=donde)
  return redent,redsal,entradas,salidas

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")

#Cargamos los datos, todos
fichdatos=open('casas.trn','r')
datos= [[float(cada) for cada in linea.strip().split()] for linea in fichdatos.readlines()]
numentradas=len(datos[0])-1

#Separamos ajuste y prueba
porazar=0.4  # 0.4
numuestras=len(datos)
muesajuste=0.8  # 0.7
muesval=0.1  # 0.15
#Desordena conjunto
random.shuffle(datos)
#Llevarlos a escala unitaria
datot=stats.zscore(array(datos,dtype=np.float32))
#Separa una parte para escoger por azar
limazar=int(porazar*numuestras)
datazar=datot[:limazar,:]
datgrup=datot[limazar:,:]

#Separa un primer lote de ajuste y prueba por azar
limajaz=int(limazar*muesajuste)
limvalaz=int(limazar*(muesajuste+muesval))
datajaz=datazar[:limajaz,:]
datvalaz=datazar[limajaz:limvalaz,:]
datpruaz=datazar[limvalaz:,:]
#Separa un segundo lote de ajuste y prueba por agrupamiento
limgrupaj=len(datgrup)
numajgrup=int(limgrupaj*muesajuste)
centros,grupos=scipy.cluster.vq.kmeans2(datgrup,numajgrup,minit='points')
orgpuntos=scipy.spatial.KDTree(datgrup)
dist,ind=orgpuntos.query(centros)
datajgrup=datgrup[ind]
indvalpru=np.setdiff1d(range(limgrupaj),ind)
datvalprugrup=datgrup[indvalpru]
numprugrup=int(limgrupaj*(1-muesval-muesajuste))
centros,grupos=scipy.cluster.vq.kmeans2(datvalprugrup,numprugrup,minit='points')
orgpuntos=scipy.spatial.KDTree(datvalprugrup)
dist,ind=orgpuntos.query(centros)
datprugrup=datvalprugrup[ind]
indpru=np.setdiff1d(range(len(datvalprugrup)),ind)
datvalgrup=datvalprugrup[indpru]

dataj=np.vstack((datajaz,datajgrup))
datval=np.vstack((datvalaz,datvalgrup))
datpru=np.vstack((datpruaz,datprugrup))

#Pasarlo a tensores torch
tea,tsa,ea,sa=septorch(dataj,dtype,device)
tev,tsv,ev,sv=septorch(datval,dtype,device)
tep,tsp,ep,sp=septorch(datpru,dtype,device)

#Si interesa la referencia de regresiÃ³n lineal, Torch no la tiene directamente, pero puedes hacer una red enteramente lineal y ajustarla


#definir red
ocultos=5
red=nn.Sequential(
    #Transforma1D(numentradas, ocultos),
    #nn.Linear(numentradas, 1)
    nn.Linear(numentradas, ocultos),
    nn.Tanh(),#Para un modelo lineal, suprimirÃ­as esto
    nn.Linear(ocultos, 1),
)

#.cuda(device)
#definir error a optimizar
error = nn.MSELoss()
#definir algoritmo de ajuste
ajuste=torch.optim.LBFGS(red.parameters(),lr=0.0005,max_iter=50,history_size=10)

nvalfal=0
def evalua():
        ajuste.zero_grad()
        s = red(tea)
        e = error(s, tsa)
        e.backward()
        return e
print ("IteraciÃ³n","Error de ajuste","Error de validaciÃ³n")
for it in range(100): # Calcula salidas
  ea=evalua()
  salval = red(tev)
  ev=error(salval,tsv)
  if 'evprevio' in dir():
    if evprevio<ev.item():
      nvalfal=nvalfal+1
    else:
      nvalfal=0
  if nvalfal>5:
    break
  evprevio=ev.item()
  print(it, math.sqrt(ea.item()),math.sqrt(evprevio))

  ajuste.step(evalua)

#Prueba: pasada directa, incluyendo derivadas de la red
ajuste.zero_grad()
salpru=red(tep)
ep=error(salpru,tsp)
print("Error de prueba",math.sqrt(ep.item()))