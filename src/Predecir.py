import csv
import math
import pickle
import numpy as np
from Neural_Network.Data import Data
from Neural_Network.Model import NN_Model

ARCHIVO_DATOS = "Dataset.csv"
ARCHIVOS_MUNI = "Municipios.csv"
class Predecir:
    def __init__(self, ruta_datos, ruta_modelo):
        self.LATITUD2      = 14.589246
        self.LONGITUD2     = -90.551449
        self.min_edad      = 0
        self.max_edad      = 0
        self.min_anio      = 0
        self.max_anio      = 0
        self.min_dist      = 0
        self.max_dist      = 0
        self.lista_muni    = None
        self.modelo        = None
        self.inicializar(ruta_datos)
        self.cargarModelo(ruta_modelo)
    
    def predecir(self, genero, edad, inscripcion, departamento, municipio):
        if self.modelo != None:
            genero_res = genero
            edad_res   = self.escalarVariable(edad, self.min_edad, self.max_edad)
            inscripcion_res = self.escalarVariable(inscripcion, self.min_anio, self.max_anio)
            distancia_res = self.escalarVariable( self.getDistancia(departamento, municipio), self.min_dist, self.max_dist)
            arreglo_x = np.array([np.array([genero_res, edad_res, inscripcion_res, distancia_res])])
            arreglo_x = arreglo_x.T
            arreglo_y = np.array([[1]])
            arreglo_y = arreglo_y.T

            val_set   = Data(arreglo_x, arreglo_y)
            #nNuevo = NN_Model(self.modelo.data, [self.modelo.data.n, 7, 7, 5, 5, 3, 1], alpha=self.modelo.alpha, iterations=self.modelo.max_iteration, lambd=self.modelo.lambd, keep_prob=self.modelo.kp)
            #nNuevo.parametros = self.modelo.parametros
            p = self.modelo.predictNormal(val_set)
            
            return p

        else:
            print("NO HAY MODELO CARGADO EN LA APLICACION")
        pass

    def inicializar(self, ruta):
        lista_muni = []
        self.lista_muni = lista_muni
        lista_training = []
        # Leer contenido del CSV
        with open(ruta + "/" + ARCHIVO_DATOS) as datos_training:
            with open(ruta + "/" + ARCHIVOS_MUNI) as datos_muni:
                reader = csv.DictReader(datos_training)
                for row in reader:
                    lista_training.append(row)
                reader2 = csv.DictReader(datos_muni)
                for row in reader2:
                    lista_muni.append(row)
        # === Manejo de datos EDAD ===
        self.min_edad = int(min(lista_training, key=lambda x: int(x['edad']))['edad'])
        self.max_edad = int(max(lista_training, key=lambda x: int(x['edad']))['edad'])

        # === Manejo de datos Inscripcion === 
        self.min_anio = int(min(lista_training, key=lambda x: int(x['Anio']))['Anio'])
        self.max_anio = int(max(lista_training, key=lambda x: int(x['Anio']))['Anio'])

        # === Manejo de datos Distancia de Municipio ===
        self.min_dist = min(lista_muni, key=lambda x: self.distancia(float(x['Lat']), float(x['Lon']), self.LATITUD2, self.LONGITUD2))
        self.min_dist = self.distancia(float(self.min_dist['Lat']), float(self.min_dist['Lon']), self.LATITUD2, self.LONGITUD2)
        self.max_dist = max(lista_muni, key=lambda x: self.distancia(float(x['Lat']), float(x['Lon']), self.LATITUD2, self.LONGITUD2))
        self.max_dist = self.distancia(float(self.max_dist['Lat']), float(self.max_dist['Lon']), self.LATITUD2, self.LONGITUD2)
    
    # Calcular distancia de municipios
    def getDistancia(self, depto, muni):
        dict = next((item for item in self.lista_muni if item["Depto"] == depto and item["Muni"] == muni), False)
        return self.distancia( float(dict['Lat']), float(dict['Lon']), self.LATITUD2, self.LONGITUD2)


    # Distancia de Coordenadas
    def distancia(self, lat1, lon1, lat2, lon2):
        rad=math.pi/180
        dlat=lat2-lat1
        dlon=lon2-lon1
        R=6372.795477598
        a=(math.sin(rad*dlat/2))**2 + math.cos(rad*lat1)*math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
        distancia=2*R*math.asin(math.sqrt(a))
        return distancia

    # Escalamiento de Variables
    def escalarVariable(self, x, xmin, xmax):
        return (x - xmin) / (xmax - xmin)

    def cargarModelo(self, ruta_modelo):
        self.modelo = pickle.load(open(ruta_modelo + ".ml", "rb"))

