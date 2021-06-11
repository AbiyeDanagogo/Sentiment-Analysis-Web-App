# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:31:35 2021

@author: MONIORH DANAGOGO
"""
import gzip
import dill
from flask import Flask, request, render_template, redirect

app = Flask(__name__)

@app.route('/')
def main():
    return redirect('/index')

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/about')
def about():
    return 'this page is all about my ML model'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    if request.method == 'GET':
        tweet = request.args.get('tweet')
    else:
        tweet = request.form['text']
    
    with gzip.open('sentimental_model.dill.gz', 'rb') as f:
        model = dill.load(f)
        
    proba = model.predict_proba([tweet])[0, 1]
    
    return 'positive sentiment: {}'.format(proba)

    

if __name__ == '__main__':
    app.run()
    
    
