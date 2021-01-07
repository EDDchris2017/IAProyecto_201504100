from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
import json
import shutil
import os
from Modelo   import *
from Predecir import Predecir


UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

# Incializar Prediccion del Modelo
predecir = Predecir("../datasets", "modelo2")

@app.route('/status')
def status():
    return "Funcionando servidor de Proyecto IA"

@app.route('/')
def home_form():
    return render_template("index.html", lista_muni=predecir.lista_muni)

@app.route('/entrenar')
def entrenar():
    return render_template("entrenar.html")

@app.route('/probar', methods=['GET', 'POST'])
def probar():
    if request.method == 'POST':
        iteraciones     = int(request.form['iteraciones'])
        numPoblacion    = int(request.form['poblacion'])
        nombreModelo    = (request.form['modelo'])
        nombreGrafica   = (request.form['grafica'])
        print("Generaciones -> ", iteraciones)
        print("Poblacion    -> ", numPoblacion)
        print("Nombre Modelo-> ", nombreModelo)
        print("Grafica      -> ", nombreGrafica)
        try:
            entrenamiento("../datasets", nombreModelo, nombreGrafica, generaciones=iteraciones, numPoblacion=numPoblacion)
            return render_template("entrenar.html", respuesta = 1)
        except:
            return render_template("entrenar.html")
    return render_template("entrenar.html")
    

@app.route('/evaluar', methods=['GET', 'POST'])
def evaluar():
    genero          = int(request.form['genero'])
    edad            = int(request.form['edad'])
    inscripcion     = int(request.form['inscripcion'])
    departamento    = request.form['departamento']
    municipio       = request.form['municipio']

    print(genero)
    print(edad)
    print(inscripcion)
    print(departamento)
    print(municipio)

    # Predecir parametro 
    res = predecir.predecir(genero, edad, inscripcion, departamento, municipio)

    return render_template("index.html", lista_muni=predecir.lista_muni, resultado=res[0][0])

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)
