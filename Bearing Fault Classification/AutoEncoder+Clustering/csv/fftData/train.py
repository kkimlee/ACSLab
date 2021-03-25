import os
import csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models, utils
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Input, Flatten, concatenate, LeakyReLU, Dropout, Reshape

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


def search(dirname, extension):
    file_list = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == extension and extension != '':
            file_list.append(full_filename)
        else:
            file_list.append(full_filename)
            
    return file_list
'''
C1_category = search('./train/weight_0', '')

C1_all = list()
C1_sp = list()
C1_un = list()
C1_un_label = list()
for data_dir in C1_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['fft']
        C1_all.append(data)
        
    C1_sp.append(pd.read_csv(C1_data[0])['fft'])
    i = 0
    for file in C1_data[1:]:
        C1_un.append(pd.read_csv(file)['fft'])
        C1_un_label.append(i)
    i += 1
    
C1_category = search('./train/weight_1', '')
for data_dir in C1_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['fft']
        C1_all.append(data)
        
    C1_sp.append(pd.read_csv(C1_data[0])['fft'])
    i = 0
    for file in C1_data[1:]:
        C1_un.append(pd.read_csv(file)['fft'])
        C1_un_label.append(i)
    i += 1

C1_category = search('./train/weight_2', '')
for data_dir in C1_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['fft']
        C1_all.append(data)
        
    C1_sp.append(pd.read_csv(C1_data[0])['fft'])
    i = 0
    for file in C1_data[1:]:
        C1_un.append(pd.read_csv(file)['fft'])
        C1_un_label.append(i)
    i += 1

C1_category = search('./train/weight_3', '')
for data_dir in C1_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['fft']
        C1_all.append(data)
        
    C1_sp.append(pd.read_csv(C1_data[0])['fft'])
    i = 0
    for file in C1_data[1:]:
        C1_un.append(pd.read_csv(file)['fft'])
        C1_un_label.append(i)
    i += 1

C1_all = np.array(C1_all)[:, :]
C1_sp = np.array(C1_sp)[:, :]
C1_un = np.array(C1_un)[:, :]
'''
encoder_input = Input(shape=(512,))
encoded = Dense(320, activation=LeakyReLU())(encoder_input)
encoded = Dense(160, activation=LeakyReLU())(encoder_input)
encoded = Dense(80, activation=LeakyReLU())(encoder_input)
encoded = Dense(128)(encoder_input)

decoded = Dense(80,  activation=LeakyReLU())(encoded)
decoded = Dense(160,  activation=LeakyReLU())(encoded)
decoded = Dense(320,  activation=LeakyReLU())(encoded)
decoded = Dense(512,  activation=LeakyReLU())(encoded)

autoencoder = Model(encoder_input, decoded)

encoder = Model(encoder_input, encoded)

decoder_input = Input(shape=(128,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(decoder_input, decoder_layer(decoder_input))

autoencoder.compile(optimizer=Adam(lr=0.0001),
              loss='mse')

autoencoder.summary()
'''
history = autoencoder.fit(C1_all, C1_all,
       batch_size=64,
       epochs=40,
       validation_data=(C1_all, C1_all))

def encoder_loss(ae_loss, inter_loss):
    def loss(y_true, y_pred):
        cluster_loss = K.mean(K.mean((y_true - y_pred)) + abs(ae_loss) - abs(inter_loss))

        return cluster_loss
    return loss
    

for epi in range(1):   
    sp_latent = encoder.predict(C1_sp)
    sp_centroid = list()
    abs_sp_centroid = list()
    for i in range(len(sp_latent)):
        abs_sp_centroid.append(abs(sp_latent[i]))
        for j in range(100):
            sp_centroid.append(sp_latent[i])
    
    inter_loss = 0
    for i in range(len(abs_sp_centroid)-1):
        inter_loss = np.mean(abs_sp_centroid[i]-abs_sp_centroid[i+1])
    inter_loss = inter_loss/(len(abs_sp_centroid)-1)
    
    sp_centroid = np.array(sp_centroid)
    sp_centorid = np.array(sp_centroid)
    
    for epoch in range(5):
        ae_loss = autoencoder.fit(C1_un, C1_un, batch_size=64, epoch=1).history['loss'][0]
        encoder.compile(optimizer=Adam(lr=0.0001),
                        loss=encoder_loss(ae_loss=ae_loss, inter_loss=inter_loss))

        encoder.fit(C1_un, sp_centroid,
                    batch_size=64,
                    epochs=1)

# classifiered_input = Input(shape=(128,))
classifiered = Dense(320, activation=LeakyReLU())(encoded)
classifiered = Dense(160, activation=LeakyReLU())(classifiered)
classifiered = Dense(128, activation=LeakyReLU())(classifiered)
classifiered = Dense(10, activation='softmax')(classifiered)

classifier = Model(encoder_input, classifiered)

classifier.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

label_encoder = LabelEncoder()

C1_un_label = label_encoder.fit_transform(np.array(C1_un_label))
C1_un_label = utils.to_categorical(C1_un_label, 10)

C1_test_category = search('./test/weight_0', '')

classifier.fit(C1_un, C1_un_label, batch_size=64, epochs=400)

C1_test = list()
C1_test_label = list()
i = 0
for data_dir in C1_test_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['DE_fft']
        C1_test.append(data)
        C1_test_label.append(i)
    i += 1

C1_test_category = search('./test/weight_1', '')
i = 0
for data_dir in C1_test_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['DE_fft']
        C1_test.append(data)
        C1_test_label.append(i)
    i += 1

C1_test_category = search('./test/weight_2', '')
i = 0
for data_dir in C1_test_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['DE_fft']
        C1_test.append(data)
        C1_test_label.append(i)
    i += 1

C1_test = np.array(C1_test)[:, :]
    
C1_test_label = label_encoder.fit_transform(np.array(C1_test_label))
C1_test_label = utils.to_categorical(C1_test_label, 10)

test_loss, test_acc = classifier.evaluate(C1_test, C1_test_label)
print(test_acc)
'''