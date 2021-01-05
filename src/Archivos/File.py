#Lectura y escritura de archivos
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import math
import os

ARCHIVO_DATOS = "Dataset.csv"
ARCHIVOS_MUNI = "Municipios.csv"

LATITUD2      = 14.589246
LONGITUD2     = -90.551449


# Carga de Datos de CSV
def cargarDatos(ruta):
    lista_training = []
    lista_muni = []
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
    min_edad = int(min(lista_training, key=lambda x: int(x['edad']))['edad'])
    max_edad = int(max(lista_training, key=lambda x: int(x['edad']))['edad'])

    # === Manejo de datos Inscripcion === 
    min_anio = int(min(lista_training, key=lambda x: int(x['Año']))['Año'])
    max_anio = int(max(lista_training, key=lambda x: int(x['Año']))['Año'])

    # === Manejo de datos Distancia de Municipio ===
    min_dist = min(lista_muni, key=lambda x: distancia(float(x['Lat']), float(x['Lon']), LATITUD2, LONGITUD2))
    min_dist = distancia(float(min_dist['Lat']), float(min_dist['Lon']), LATITUD2, LONGITUD2)
    max_dist = max(lista_muni, key=lambda x: distancia(float(x['Lat']), float(x['Lon']), LATITUD2, LONGITUD2))
    max_dist = distancia(float(max_dist['Lat']), float(max_dist['Lon']), LATITUD2, LONGITUD2)

    # === Obtener arreglo de datos de entrenamiento ===
    arreglo_training = np.array([])
    for element in lista_training:
        genero = 0
        if element['Genero'] == "MASCULINO" : genero = 1
        dist = escalarVariable( getDistancia(element['cod_depto'], element['cod_muni'], lista_muni), min_dist, max_dist)
        edad = escalarVariable(int(element['edad']), min_edad, max_edad)
        anio = escalarVariable(int(element["Año"]), min_anio, max_anio)
        salida = 0
        if element['Estado'] == "Activo": salida = 1
        arreglo_training = np.append(arreglo_training, {"datos": np.array([genero,edad,anio,dist]), "salida" : salida})
    
    # === Obtener Datos de entrenamiento y Validacion ===
    np.random.shuffle(arreglo_training)
    cant_training = int(len(arreglo_training) * 80 / 100)
    # Datos Training
    result_trainig = arreglo_training[0 : cant_training]
    train_X = np.array([o['datos']  for o in result_trainig])
    train_X = train_X.T
    train_Y = np.array([[o['salida'] for o in result_trainig]])
    # Datos Validacion
    result_val = arreglo_training[cant_training :]
    val_X = np.array([o['datos']  for o in result_val])
    val_X = val_X.T
    val_Y = np.array([[o['salida'] for o in result_val]])

    # Configurar resultados
    print(train_X.shape)
    print(train_Y.shape)
    print(val_X.shape)
    print(val_Y.shape)
    return train_X, train_Y, val_X, val_Y 

    
def escalarVariable(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

# Calcular distancia de municipios
def getDistancia(depto, muni, lista_muni):
    dict = next((item for item in lista_muni if item["Depto"] == depto and item["Muni"] == muni), False)
    return distancia( float(dict['Lat']), float(dict['Lon']), LATITUD2, LONGITUD2)


# Distancia de Coordenadas
def distancia(lat1, lon1, lat2, lon2):
    rad=math.pi/180
    dlat=lat2-lat1
    dlon=lon2-lon1
    R=6372.795477598
    a=(math.sin(rad*dlat/2))**2 + math.cos(rad*lat1)*math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
    distancia=2*R*math.asin(math.sqrt(a))
    return distancia

#cargarDatos("../../datasets")