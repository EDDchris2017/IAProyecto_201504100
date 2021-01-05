# Generar Modelo de Prediccion
from Archivos import File
from Neural_Network.Data import Data
from Neural_Network.Model import NN_Model
from Util import Plotter

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

entrenarModelo()
