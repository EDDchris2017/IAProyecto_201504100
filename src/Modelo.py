# Generar Modelo de Prediccion
from Archivos import File
from Neural_Network.Data import Data
from Neural_Network.Model import NN_Model
from Util import Plotter
from Genetico import Algoritmo 
import pandas as pd
from Genetico.Algoritmo import *
import pickle



def entrenamiento(ruta, nombre_modelo, nombre_grafica, generaciones, numPoblacion):
    # comenzar entrenamiento con Algoritmo Genetico
    params = comenzarGenetico(generaciones, ruta, "principal", numPoblacion)
    params_solucion = params['params']

    # Imprimir Datos
    print("alpha: ", params_solucion['alpha'])
    print("iterations: ", params_solucion['iterations'])
    print("lambd: ", params_solucion['lambd'])
    print("keep_prob: ", params_solucion['keep_prob'])

    # Se define el modelo
    nn1 = NN_Model(params['train_set'], params['capas'], alpha=params_solucion['alpha'], iterations=params_solucion['iterations'], lambd=params_solucion['lambd'], keep_prob=params_solucion['keep_prob'])
    # Se entrena el modelo
    nn1.training(False)
    Plotter.show_Model([nn1])
    # Guardar Modelo
    guardarModelo(nn1, nombre_modelo)

def guardarModelo(modelo, nombre):
    pickle.dump(modelo, open(nombre + ".ml", "wb"))



def comenzarGenetico(iteraciones, ruta, resArchivo, numPoblacion):
    # Lectura archivo de Entrenamiento y Validacion
    train_X, train_Y, val_X, val_Y = File.cargarDatos(ruta)

    # ====== DEFINICION DE RED NEURONAL ======
    # Definir los conjuntos de datos
    train_set = Data(train_X, train_Y)
    val_set   = Data(val_X, val_Y)

    # Se define las dimensiones de las capas
    capas = [train_set.n, 7, 7, 5, 5, 3, 1]

    #  ==== Lectura archivo CSV Hiperparametros ====
    entradas_params = {}
    df = pd.read_csv(ruta + "/" + "hiperparams.csv")
    entradas_params["alpha"]            = df['alpha'].tolist()
    entradas_params["lambda"]           = df["lambda"].tolist()
    entradas_params["max_iteration"]    = df["max_iteration"].tolist()
    entradas_params["keep_prob"]        = df["keep_prob"].tolist()

    # Inicializacion de Algoritmo Genetico
    print("Ejecutando Algoritmo Genetico ...")
    genetico = Genetico(iteraciones, entradas_params, numPoblacion, len(entradas_params["alpha"]), train_set, val_set, capas, 6)
    hiper_params = genetico.ejecutar()

    return {"params": hiper_params, "train_set" : train_set , "val_set" : val_set, "capas" : capas }

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

# =================================== PREDICCION DE DATOS ==================================
def prediccionDatos(genero, edad, inscripcion, departamento, municipio):
    respuesta = 0

    return respuesta


#entrenamiento("../datasets", "modelo", "graficaModelo", generaciones=50, numPoblacion=10)
entrenamiento("../datasets", "modelo2", "graficaModelo", generaciones=50, numPoblacion=10)
