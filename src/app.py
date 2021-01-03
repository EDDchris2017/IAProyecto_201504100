from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
import json
import shutil
import os

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)


@app.route('/status')
def status():
    return "Funcionando servidor de Proyecto IA"

@app.route('/')
def home_form():
    return render_template("index.html")

@app.route('/evaluar')
def evaluar():
    return render_template("index.html")

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)
