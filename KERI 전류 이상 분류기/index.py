import os
import csv
import json
import numpy as np
import pandas as pd

import librosa
import librosa.display

from flask import Flask, render_template, request, jsonify

from tensorflow.keras.models import load_model
app = Flask(__name__)

data_path = 'data/'

file=''
data = ''
category = 'Va'
index = []
value = []

start = 0
end = 0
batch = 0
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', data=[0, 0, 0, 0])

@app.route('/getData', methods=['GET', 'POST'])
def getData():
    global category
    global file
    global data
    global index
    global value
    global start
    global end
    global batch
    
    start = 0
    end = 0
    batch = 0
    
    if request.method == 'POST':
        print(request.form.to_dict())
        key = list(request.form.to_dict().keys())
        
        if key[0] == 'file':
            file = request.form[key[0]]
            
        elif key[0] == 'category':
            category = request.form[key[0]]
    
    if file != '':
        data = pd.read_csv(data_path+file+'.csv')[category]
        if len(data) > 500000:
            start = 503000
            batch = 50000
            
        else:
            batch = len(data)//10
        end = start + batch
        
        tmp_data = data[start:end]
        json_data = tmp_data.to_json(orient='split')
        json_data = json.loads(json_data)
        index = json_data['index']
        value = json_data['data']
        json_data = {'category':category, 'index':index, 'value':value}

    return json_data

@app.route('/movePage', methods=['GET', 'POST'])
def movePage():
    global category
    global file
    global data
    global index
    global value
    global start
    global end
    global batch
     
    if request.method == 'POST':
        key = list(request.form.to_dict().keys())
        
        if request.form[key[0]] == 'next':
            if start < len(data) - batch:
                start += batch
                end = start + batch
            else:
                start = start
                end = start + batch
        elif request.form[key[0]] == 'previous':
            if start >= batch:
                start -= batch
                end = start + batch
            else:
                start = start
                end = start + batch
    
    tmp_data = data[start:end]
    json_data = tmp_data.to_json(orient='split')
    json_data = json.loads(json_data)
    index = json_data['index']
    value = json_data['data']
    json_data = {'category':category, 'index':index, 'value':value}
   
    return json_data
    

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    global category
    global data
    global index
    global value
    global start
    global end

    model = load_model('data/model_Vc')
    
    spectrogram = []
    result = []
    
    tmp_data = data[start:end]    
    for i in range(len(tmp_data)//10000):
        spectrogram.append(np.array(np.abs(librosa.stft(np.array(tmp_data[i*10000:(i+1) * 10000])))))
    spectrogram = np.array(spectrogram)
    spectrogram = np.reshape(spectrogram, spectrogram.shape + (1,))
    
        
    for i in range(len(spectrogram)):
        result.append(np.int16(np.argmax(model.predict(spectrogram)[i])).item())
        
    json_data = {'category':category, 'index':index, 'value':value, 'result':result}
    
    return json_data

if __name__ == '__main__':
    app.run(host="0.0.0.0")