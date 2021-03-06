import csv
import numpy as np
import pandas as pd
import random
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Neural_Network')))
from Neural_Network.Model import NN_Model



class Nodo:
    def __init__(self, solucion, fitness):
        self.solucion = solucion
        self.fitness  = fitness
    
    def __gt__(self, nodo):
        return self.fitness < nodo.fitness

class Genetico:

    def __init__(self, iteraciones, entrada_training, numPoblacion, rangoDatos, train_set, val_set, capas, capasN):
        self.iteraciones  = iteraciones
        self.entrada_training      = entrada_training
        self.numPoblacion = numPoblacion
        self.rangoDatos   = rangoDatos
        self.train_set    = train_set
        self.val_set      = val_set
        self.capas        = capas
        self.capasN       = capasN
        self.porcentaje_padres = 40
    
    def ejecutar(self):

        # Desarrollo del Algoritmo
        generacion = 0
        poblacion  = self.inicializarPoblacion()
        fin = self.verificarCriterio(poblacion, generacion)
        self.imprimirPoblacion(poblacion, generacion)
        while fin == None:
            padres      = self.seleccionarPadres( poblacion)
            poblacion   = self.emparejar(padres)
            generacion  += 1
            fin = self.verificarCriterio( poblacion, generacion)
            self.imprimirPoblacion(poblacion, generacion)

        # Guardar en bitacora 
        print("Termino el algoritmo ...")
        print(fin.solucion)

        hiperparametros = self.getDataTraining(fin.solucion[0], fin.solucion[1], fin.solucion[2], fin.solucion[3])
        try:
            self.guardarBitacora(fin, hiperparametros)
        except:
            print("Error al guardar en bitacora")

        return hiperparametros
    
    def imprimirPoblacion(self, poblacion, generacion):
        maximo = max(poblacion, key=lambda x: x.fitness)
        print("-> Generacion ", generacion, " : ", maximo.fitness)
        print("     ",maximo.solucion)
    
    # {'alpha' : alpha, 'lambd' : lambd, 'iterations' : max_iteration, 'keep_prob' : keep_prob}
    def guardarBitacora(self, fin, params):
        cad_registro = ""
        cad_registro += str(fin.fitness) + "\n"
        cad_registro += str(fin.solucion) + "\n"
        cad_registro += "alpha: " + str(params['alpha']) + "\n"
        cad_registro += "iterations: " + str(params['iterations']) + "\n"
        cad_registro += "lambd: " + str(params['lambd']) + "\n"
        cad_registro += "keep_prob: " + str(params['keep_prob']) + "\n"

        f = open("log" + ".txt", 'a')
        f.write('\n' + cad_registro + '\n')
        f.close()

    # ===================================== INICIALIZAR POBLACION =====================================
    def inicializarPoblacion(self):
        poblacion = []
        for i in range(self.numPoblacion):
            poblacion.append(self.crearIndividuo())

        return poblacion
    
    def crearIndividuo(self):
        solucion = []
        for i in range(4):
            solucion.append(random.randint(0, self.rangoDatos - 1)) # Rango de datos aleatorios de Individuo
        nuevo_nodo = Nodo(
            solucion,
            self. evaluarFitness(solucion)
        )
        return nuevo_nodo
    
    def evaluarFitness(self, solucion):
        # Recorrer contenido del archivo
        fitness = 0
        params_solucion = self.getDataTraining(solucion[0], solucion[1], solucion[2], solucion[3])

        # Se define el modelo
        nn1 = NN_Model(self.train_set, self.capas, alpha=params_solucion['alpha'], iterations=params_solucion['iterations'], lambd=params_solucion['lambd'], keep_prob=params_solucion['keep_prob'])

        # Se entrena el modelo
        nn1.training(False)

        # Validacion Modelo
        fitness = nn1.predict(self.val_set)

        return fitness
    
    def getDataTraining(self, alpha_n, lambda_n, max_iteration_n, keep_prob_n):
        alpha  = float(self.entrada_training['alpha'][alpha_n])
        lambd  = float(self.entrada_training['lambda'][lambda_n])
        max_iteration = int(self.entrada_training['max_iteration'][max_iteration_n])
        keep_prob = float(self.entrada_training['keep_prob'][keep_prob_n])

        return {'alpha' : alpha, 'lambd' : lambd, 'iterations' : max_iteration, 'keep_prob' : keep_prob}

    
    # ===================================== VERIFICACION FINALIZACION =====================================
    def verificarCriterio(self, poblacion, generacion):
        fin = None
        if generacion == self.iteraciones:
            fin = max(poblacion, key=lambda x: x.fitness)
        return fin
    
    # ===================================== SELECCION DE PADRES =====================================
    # Tomar Porcentaje de padres
    def seleccionarPadres(self, poblacion):
        mejores_padres = []
        padres_considerar = int(self.porcentaje_padres * len(poblacion) / 100)
        mejores_padres = sorted(poblacion)[:padres_considerar]
        mejores_padres = mejores_padres.copy()
        return mejores_padres
    
    # ===================================== EMPAREJAMIENTO DE PADRES ===================================== 
    def emparejar(self, padres):
        nueva_poblacion = padres.copy()
        print("PADRES ", len(nueva_poblacion))
        cont = 0
        # Recorrer padres
        while cont < self.numPoblacion - len(padres):
            for pos1 in range(len(padres)):
                padre1 = padres[pos1]
                for pos2 in range(pos1 + 1, len(padres)):
                    padre2 = padres[pos2]
                    nueva_poblacion.append( self.cruzar(padre1, padre2, 1))
                    cont += 1
                    if cont == self.numPoblacion - len(padres) :
                        return nueva_poblacion
        return nueva_poblacion
    
    def cruzar(self, p1, p2, tipo):
        nuevo_hijo = [None] * 4
        padre1 = p1.solucion
        padre2 = p2.solucion
        for i in range(4):
            actual = padre1
            esP = random.randint(1,2)
            if esP == 1:
                actual = padre1
            else:
                actual = padre2
            nuevo_hijo[i] = actual[i]
        nuevo_hijo = self.mutar(nuevo_hijo)
        hijoNodo   = Nodo(
            nuevo_hijo,
            self.evaluarFitness(nuevo_hijo)
        )
        return hijoNodo

    def mutar(self, solucion):
        mutado = solucion
        pos = random.randint(0,3)
        val = random.randint(0, self.rangoDatos - 1)
        mutado[pos] = val

        return mutado
