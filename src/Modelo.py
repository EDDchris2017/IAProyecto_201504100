# Generar Modelo de Prediccion
from Archivos import File
from Neural_Network.Data import Data
from Neural_Network.Model import NN_Model
from Util import Plotter
from Genetico import Algoritmo 
import pandas as pd
from Genetico.Algoritmo import *



def entrenamiento(ruta):
    # comenzar entrenamiento con Algoritmo Genetico
    comenzarGenetico(2, ruta, "principal")


def comenzarGenetico(iteraciones, ruta, resArchivo):
    hiper_params = None
    # Lectura archivo de Entrenamiento y Validacion
    train_X, train_Y, val_X, val_Y = File.cargarDatos(ruta)

    # ====== DEFINICION DE RED NEURONAL ======
    # Definir los conjuntos de datos
    train_set = Data(train_X, train_Y)
    val_set   = Data(val_X, val_Y)

    # Se define las dimensiones de las capas
    capas = [train_set.n, 7, 5, 5, 3, 1]

    #  ==== Lectura archivo CSV Hiperparametros ====
    entradas_params = {}
    df = pd.read_csv(ruta + "/" + "hiperparams.csv")
    entradas_params["alpha"]            = df['alpha'].tolist()
    entradas_params["lambda"]           = df["lambda"].tolist()
    entradas_params["max_iteration"]    = df["max_iteration"].tolist()
    entradas_params["keep_prob"]        = df["keep_prob"].tolist()

    # Inicializacion de Algoritmo Genetico
    print("Ejecutando Algoritmo Genetico ...")
    genetico = Genetico(iteraciones, entradas_params, 16, len(entradas_params["alpha"]), train_set, val_set, capas, 6)
    genetico.ejecutar()

    return hiper_params

def entrenarModelo():
    # Cargando conjunto de datos
    train_X, train_Y, val_X, val_Y = File.cargarDatos("../datasets")

    #Definir los conjuntos de datos
    train_set = Data(train_X, train_Y)
    val_set   = Data(val_X, val_Y)

    # Se define las dimensiones de las capas
    capas1 = [train_set.n, 10, 5, 1]

    # Se define el modelo
    nn1 = NN_Model(train_set, capas1, alpha=0.001, iterations=50000, lambd=0, keep_prob=0.5)

    # Se entrena el modelo
    nn1.training(False)

    # Se analiza el entrenamiento
    Plotter.show_Model([nn1])

    print('Entrenamiento Modelo 1')
    nn1.predict(train_set)
    print('Validacion Modelo 1')
    nn1.predict(val_set)

entrenamiento("../datasets")
