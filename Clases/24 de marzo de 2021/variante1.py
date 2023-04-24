#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

argums = argparse.ArgumentParser(description="""
Red con una capa convolutiva normal de regresiÃ³n, otra con varios anchos y otra normal usando pytorch 
Realiza varios anÃ¡lisis someros
""", epilog="""
Ejemplo:
variante1 -fichdatos imorig -cpu -minibatch 100 -limit 1000
"""
                                 )
argums.add_argument('-fichdatos', type=argparse.FileType('rb'), nargs=1, required=True,
                    help='fichero con los datos de imÃ¡genes')
argums.add_argument('-minibatch', type=int, nargs='?', default=1,
                    help='numero de casos usado para ajuste', )
argums.add_argument('-paciencia', type=int, nargs='?', default=0,
                    help='paciencia para cambiar tasa de aprendizaje')
argums.add_argument('-dectasa', type=float, nargs='?', default=0,
                    help='si mayor que 0, factor de decrecimiento de tasa de aprendizaje')
argums.add_argument('-momento', type=float, nargs='?', default=0,
                    help='momento')
argums.add_argument('-limit', type=int, nargs=1, required=True,
                    help='límite de iteraciones de cada ajuste')
procesador = argums.add_mutually_exclusive_group()
procesador.add_argument('-cpu', dest='cpucuda', action='store_const', const='cpu',
                        help='usar nÃºcleos de CPU')


class Opcioncuda(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, "cuda:%r" % values[0])


procesador.add_argument('-cuda', type=int, action=Opcioncuda, nargs=1, dest='cpucuda', metavar='',
                        help='usar coprocesador de tarjeta grÃ¡fica de NVIDIA')
argums.add_argument('-fichpesos', type=argparse.FileType('wb'), nargs=1, required=True,
                    help='fichero para guardar pesos de red completa')

import copy
from numpy import array
import random
import sys
import psutil
import time
import torch
import torch.nn as nn
import signal
import traceback
from PIL import Image
import math


def ereg(error):
    """
FunciÃ³n que devuelve una medida de error (RMSE) dado un objeto de error de Pytorch
    """
    return math.sqrt(error.item())


def menser(error):
    """
Sacar la precisiÃ³n a partir del error (complementario)
>>>menser(0.1)
PrecisiÃ³n=0.9
    """
    return "Precisión={}".format(1 - error)


def mostrar(red):
    """
Mostrar un caso al azar: imagen de entrada, salida ideal y salida de la red
    """
    global datent, salidas, numuestras
    i1 = random.randint(0, numuestras - 1)
    datos = datent[i1:i1 + 1, :, :, :]
    salred = red(datos)
    imagorig = Image.fromarray(datos.detach().numpy().squeeze() * 255)
    imagorig.show()
    print("Real: {}".format(salidas[i1]), 'Red: {}'.format(salred.detach().numpy()))


def ejemplos(red):
    """
 Control del ciclo de ir mostrando ejemplos con una red
    """
    seguir = "s"
    while (seguir == "s"):
        mostrar(red)
        seguir = input("Teclea s si quieres más ")


def procesacapas(red, previo, sini):
    """
  Agrupar las descripciones y salidas de las capas que nos interesan
    """
    global descripcapa
    salcapa = []
    for capa, guardar, descrip in zip(red.children(), sini, descripcapa):
        previo = capa(previo)
        if guardar:
            salcapa.append((descrip, previo))
    return salcapa


def analmin(tep, red):
    """
Ejemplo de pequeÃ±o anÃ¡lisis del comportamiento de una red con un conjunto de prueba
Saca quÃ© procesadores varÃ­an poco con los datos
    """
    previo = tep
    salcapa = procesacapas(red, previo, salidaproc)
    for capa in salcapa:
        # Sacar varianza y mostrar los que la tengan pequeÃ±a
        varian = capa[1].cpu().detach().numpy().var(0)
        inutiles = varian < 0.1
        if len(varian[inutiles]) > 0:
            print("Capa: ", capa[0], "Procesadores poco activos: ", inutiles)


def controlsenhal(numsenhal, stackiframe):
    """
Controlador de seÃ±ales de interrupciÃ³n, segÃºn lo indicado en lista
    """
    global lista
    global it, inired, evprevio
    print("Detectada señal ", numsenhal)
    for caso in lista:
        if lista[caso]['senhal'] == numsenhal:
            sys.stdout.flush()
            sys.stderr.flush()
            lista[caso]['soltar'] = True
    if lista['inmediato']['soltar'] or lista['informacion']['soltar']:
        # Sacar informaciÃ³n de ejecuciÃ³n
        traceback.print_stack(stackiframe)
        print("IteraciÃ³n ", it, " con error de validación ", evprevio, " en prueba de optimización ", inired)
        sys.stdout.flush()
        sys.stderr.flush()
    if lista['inmediato']['soltar']:
        quit()


def chequeacarga(situacion):
    """
ComprobaciÃ³n de que no nos pasamos ni de tiempo ni de memoria
    """
    global lista
    tiempo = time.time()
    parada = False
    # ComprobaciÃ³n de tiempo
    tlim = lista[situacion]['tiempo']
    for caso in filter(lambda c: lista[c]['tiempo'] >= tlim, lista):
        if tiempo - tini > lista[caso]['tiempo']:
            lista[caso]['soltar'] = True
            lista[caso]['tiempo'] = lista[caso]['tiempo'] * 10
            print("Lleva ", tiempo - tini)
        if lista[caso]['soltar']:
            parada = True
    # ComprobaciÃ³n de memoria
    memoria = psutil.swap_memory()
    if memoria.percent > 10:
        trafico = memoria.sin + memoria.sout
        try:
            inctrafico = trafico - chequeacarga.trafprevio
            if inctrafico / memoria.total > 0.01:
                print("Detectado tráfico\nPorcentaje de swap ", memoria.percent)
                print("Memoria ", psutil.virtual_memory())
                lista['todo']['soltar'] = True
                parada = True
            elif memoria.percent > 20:
                print("Exceso de swap ", memoria.percent)
                print("Memoria ", psutil.virtual_memory())
                lista['todo']['soltar'] = True
                parada = True
        except Exception as previo:
            print("Detectado swap")
        chequeacarga.trafprevio = trafico
    return parada


def inicial(capa):  # Inicializa cada capa
    try:
        capa.reset_parameters()
    except AttributeError:
        pass


def optiauxpar(red0, limajaz, inda, indv, indp):
    """
   Pasada de optimizaciÃ³n
Recibe una red, la cantidad de casos de ajuste, sus Ã­ndices, los de validaciÃ³n y los de prueba
Devuelve el error final y la red ajustada
    """
    global datent
    global salidas
    global device, para
    global it, evprevio, args
    if proc == "cpu":
        tea = datent[inda]
        tsa = salidas[inda]
        tev = datent[indv]
        tsv = salidas[indv]
        red = copy.deepcopy(red0)
    else:
        tea = datent[inda].cuda(device)
        tsa = salidas[inda].cuda(device)
        tev = datent[indv].cuda(device)
        tsv = salidas[indv].cuda(device)
        red = copy.deepcopy(red0).cuda(device)
    red.apply(inicial)
    # definir error a optimizar
    error = nn.MSELoss()
    # error = nn.BCELoss()
    # error = nn.SmoothL1Loss()
    # error = nn.L1Loss()

    # definir algoritmo de ajuste
    # ajuste=torch.optim.LBFGS(red.parameters(),lr=0.0001,max_iter=10,history_size=80)
    # ajuste=torch.optim.Adadelta(red.parameters(),weight_decay=...)
    # ajuste=torch.optim.Adagrad(red.parameters(),lr=...,weight_decay=0.005)
    ajuste = torch.optim.Adam(red.parameters(), lr=0.001, weight_decay=0.00)  # 3)
    # ajuste=torch.optim.ASGD(red.parameters(),lr=...,weight_decay=0.004)
    # ajuste=torch.optim.RMSprop(red.parameters(),lr=...,weight_decay=0.001)
    # ajuste=torch.optim.Rprop(red.parameters(),lr=...)
    # ajuste = torch.optim.SGD(red.parameters(), lr=0.001, momentum=0.02, weight_decay=0.004, nesterov=True)
    if (args.paciencia > 0):
        ajustepaso = torch.optim.lr_scheduler.ReduceLROnPlateau(ajuste, patience=args.paciencia, factor=args.dectasa)
    #    ajustepaso=torch.optim.lr_scheduler.StepLR(ajuste,step_size=100,gamma=0.8)
    ajustemer = torch.optim.lr_scheduler.MultiplicativeLR(ajuste, lr_lambda=lambda paso: 1)
    npasadas = 1
    # 2 si se quiere segunda pasada
    limalg = args.limit[0] // npasadas
    try:
        for naj in range(npasadas):
            evprevio = 1e9
            nvalfal = 0
            numcasos = args.minibatch
            print(limalg, naj)
            for it in range(int(limalg * naj), int(limalg * (naj + 1))):
                numcasos = min(numcasos, limajaz)
                numbloques = limajaz // numcasos
                limites = [(cual * numcasos, (cual + 1) * numcasos) for cual in range(numbloques)]
                limites[-1] = (limajaz - numcasos, limajaz);
                for bloq in limites:
                    ajuste.zero_grad()
                    sa = red(tea[bloq[0]:bloq[1], :])
                    ea = error(sa.reshape(tsa[bloq[0]:bloq[1]].shape), tsa[bloq[0]:bloq[1]])
                    if math.isnan(ea.item()):
                        red.apply(inicial)
                        ajustemer.step()
                        print("Divergencia en iteración", it)
                    else:
                        ea.backward()
                        ajuste.step()
                ajuste.zero_grad()
                salval = red(tev)
                ev = error(salval.reshape(tsv.shape), tsv)
                if evprevio < ev.item():
                    nvalfal = nvalfal + 1
                else:
                    nvalfal = 0
                    evprevio = ev.item()
                print('Iteración', it, "Error de validación", ev.item())
                if (args.paciencia > 0):
                    ajustepaso.step(ev)
                salida = chequeacarga('ajuste')
                if nvalfal > 10:
                    break
                if salida:
                   break
    #            ajuste = torch.optim.SGD(red.parameters(), lr=..., momentum=mom, weight_decay=..., nesterov=True)
    except Exception as prob:
        print()
        print(prob)
        print("Terminando en iteración", it)
    red.zero_grad()
    del tea, tsa, tev, tsv
    # Prueba
    if proc == "cpu":
        tep = datent[indp].requires_grad_()
        tsp = salidas[indp]
    else:
        tep = datent[indp].cuda(device).requires_grad_()
        tsp = salidas[indp].cuda(device)
    salpru = red(tep)
    ep = error(salpru.reshape(tsp.shape), tsp)
    tfall = ereg(ep)
    print("Secuencia de optimización con ", menser(tfall), " en", it, "iteraciones")
    if salida:
        sys.stdout.flush()
        para[lista['ajuste']]['soltar'] = False
    return float(tfall), red.cpu()


def optiserie(red0, limajaz, inda, indv, indp):
    """
   Lanza varios intentos de oprimizaciÃ³n a partir de una red base
Recibe una red, la cantidad de casos de ajuste, sus Ã­ndices, los de validaciÃ³n y los de prueba
Devuelve el conjunto de errores finales y redes ajustadas
    """
    global para, lista
    resuldat = []
    global inired
    resulredes = []
    ninten = nintentos
    for inired in range(ninten):
        tfall, red = optiauxpar(red0, limajaz, inda, indv, indp)
        resuldat.append(tfall)
        resulredes.append(red)
        salida = chequeacarga('inicio')
        if salida:
            para[lista['inicio']]['soltar'] = False
            break
    return resuldat, resulredes


def pruebadatos(indices, limajaz, limvalaz, red0):
    """
   Lanza una optimizaciÃ³n de varios intentos, pero todos con los mismos datos
Recibe una permutaciÃ³n de los Ã­ndices de datos, la cantidad de casos de ajuste, el total con los de validaciÃ³n y la red
Devuelve la red de menor error y su error
    """
    global datent
    global salidas
    global lista, para
    indices = array(indices)
    inda = indices[:limajaz]
    indv = indices[limajaz:limvalaz]
    indp = indices[limvalaz:]
    parada = chequeacarga('muestra')
    if parada:
        sys.stdout.flush()
        para[lista['muestra']]['soltar'] = False
        print("Partición de datos cancelada")
        return False, False
    else:
        resuldat, resulredes = optiserie(red0, limajaz, inda, indv, indp)
        min_index, min_value = min(enumerate(resuldat), key=lambda p: p[1])
        print("Partición de datos con ", menser(min_value))
        return resulredes[min_index], min_value


def unaprueba(argumentos):
    """
   lanza pruebadatos: varios ajustes lanzados sobre el mismo conjunto de datos de entrenamiento
Recibe una permutaciÃ³n de los Ã­ndices de datos, la cantidad de casos y la red
Devuelve la red de menor error y su error
    """
    indices, numuestras, red = argumentos
    limajaz = int(muesajuste * numuestras)
    print("Partición de ajuste con", limajaz, "casos")
    limvalaz = int(numuestras * (muesajuste + muesval))
    redaj, errores = pruebadatos(indices, limajaz, limvalaz, red)
    return redaj, errores


def multiprueba(red):
    """
   lanza varios intentos de red, variando el conjunto de datos
Recibe una red
Devuelve el conjunto de errores finales y redes ajustadas para cada particiÃ³n de los datos
    """
    indices = list(range(numuestras))
    pruebas = []
    for pru in range(numpruebas):
        random.shuffle(indices)
        pruebas = pruebas + [(copy.copy(indices), numuestras, red)]
    lisresul = map(unaprueba, pruebas)
    return lisresul


class Media(nn.Module):
    """
MÃ³dulo consistente en varias cadenas de capas en paralelo, cuyas salidas deben tener las mismas dimensiones, porque se concatenan
    """

    def __init__(self, ramas):
        # ramas son los bloques separados que actÃºan
        # cada bloque es a su vez una secuencia
        super(Media, self).__init__()
        self.ramas = ramas

    def forward(self, x):
        # cada rama opera independientemente sobre el total y luego se concatenan sus resultados
        resuls = list()
        for bloque in self.ramas:
            resuls.append(bloque(x))
        return torch.cat(resuls, 1)


def promedia(lisresul):
    """
Recibe el conjunto de errores finales y redes ajustadas para cada particiÃ³n de los datos
Devuelve la mejor o FALSE si no ha podido sacar ninguna
    """
    ert = 0
    n = 0
    mejor = [0, 1e9]
    for r in lisresul:
        try:
            if r[1] < mejor[1]:
                mejor = r
            ert = ert + r[1]
            n += 1
        except Exception as e:
            print(e)
            print(r)
    if n > 0:
        print("Promedio de ", menser(ert / n))
        print("Mejor: ", mejor[1])
        torch.save(mejor[0].state_dict(), args.fichpesos[0])
        return mejor[0]
    else:
        print(lisresul)
        return False


if __name__ == '__main__':
    tini = time.time()

    # Para cada fase, su seÃ±al y su tiempo
    lista = {
        'informacion': {'senhal': signal.SIGHUP, 'tiempo': 25000},
        'ajuste': {'senhal': signal.SIGUSR1, 'tiempo': 50000},
        'inicio': {'senhal': signal.SIGUSR2, 'tiempo': 100000},
        'muestra': {'senhal': signal.SIGTSTP, 'tiempo': 150000},
        'todo': {'senhal': signal.SIGINT, 'tiempo': 200000},
        'inmediato': {'senhal': signal.SIGTERM, 'tiempo': 250000},
    }
    # Inicializar controlador y parÃ¡metro de terminaciÃ³n
    for caso in lista:
        signal.signal(lista[caso]['senhal'], controlsenhal)
        lista[caso]['soltar'] = False

    # Recoger invocaciÃ³n
    args = argums.parse_args()

    dtype = torch.float
    proc = args.cpucuda or 'cpu'  # "cpu"#"cuda:0"
    print("Ejecutando en", proc)
    device = torch.device(proc)

    numpruebas = 5  # de particiÃ³n de muestra
    nintentos = 1  # de optimizaciÃ³n
    muesajuste = 0.7
    muesval = 0.2
    datent, salidas = torch.load(args.fichdatos[0])
    # El profesor preguntÃ³ "Â¿Y cÃ³mo diablos guardaste esos datos?"
    # Y la AlumnaQueTodoLoSabÃ­a contestÃ³ "AsÃ­
    #    torch.save((datent,salidas),nombredelfichero)

    numuestras = datent.size()[0]

    r1 = 5
    a = 20
    anc1 = 4
    b1 = 10
    anc2_1 = 5
    b2 = 4
    b3 = 6
    anc2_2 = 3
    anc3 = 3
    c = 10
    r2 = 5
    d = 880

    red = nn.Sequential(
        # anchoxalto 240x320x1
        nn.AvgPool2d(r1),
        # L_out=>ancho/r1xalto/r1x1
        nn.Conv2d(1, a, anc1),
        # L_out-10=>ancho1-anc1+1xalto1-anc1+1xa
        nn.LeakyReLU(),
        Media(nn.ModuleList([
            ##############Tres ramas para intermedia. Rellenan para mantener el tamaÃ±o
            nn.Sequential(
                # ...xa
                nn.Conv2d(a, 1, 1),  # reductora
                nn.LeakyReLU(),  # no lineal
                # ...x1
                nn.Conv2d(1, b1, anc2_1, padding=2, padding_mode='replicate'),  # convolutiva
                # ...xb1
                nn.LeakyReLU(),  # no lineal
            ),
            nn.Sequential(
                # ...xa
                nn.Conv2d(a, b2, 1),  # reductora
                # ...xb2
                nn.LeakyReLU(),  # no lineal
            ),
            nn.Sequential(
                # ...xa
                nn.Conv2d(a, 1, 1),  # reductora
                nn.LeakyReLU(),  # no lineal
                # ...x1
                nn.Conv2d(1, b3, anc2_2, padding=1, padding_mode='replicate'),  # convolutiva
                # ...xb3
                nn.LeakyReLU(),  # no lineal
                nn.Hardtanh(),  # Hardtanh
            )]
        )),
        # ancho2xalto2xb1+b2+b3
        nn.Conv2d(b1 + b2 + b3, c, anc3),
        # L_out=>ancho2-anc3+1xalto2-anc3+1xc
        nn.LeakyReLU(),
        nn.AvgPool2d(r2),
        # L_out6=>ancho3/r2xalto3/r2xc
        nn.Flatten(),
        # ancho4Â·alto4Â·c=d
        nn.Linear(d, 1),
        nn.Hardtanh()
    ).cpu()
    salidaproc = [False, False, True, True, False, True, False, False, False]
    descripcapa = ['', 'l', 'Convolución 1', 'Conjunto de varias convoluciones', '', 'Convolución final', '', '', '']
    try:
        lisresul = multiprueba(red)
    except Exception as problema:
        print(problema)
        traceback.print_exc()
        probmens = problema.message.upper()
        if probmens.find('CUDA') >= 0 and probmens.find('MEMORY') >= 0:
            print("Pasando a cpu")
            proc = "cpu"
            device = torch.device(proc)
            lisresul = multiprueba(red)
    redfin = promedia(lisresul)
    if redfin:
        #         analmin(datent,redfin)
        ejemplos(redfin)

