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
predecir = Predecir("../datasets", "modelo")

@app.route('/status')
def status():
    return "Funcionando servidor de Proyecto IA"

@app.route('/')
def home_form():
    datos = 2
    return render_template("index.html", lista_muni=predecir.lista_muni)

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

    #respuesta = prediccionDatos(genero, edad, inscripcion, departamento, municipio)

    return render_template("index.html", lista_muni=predecir.lista_muni)

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)
