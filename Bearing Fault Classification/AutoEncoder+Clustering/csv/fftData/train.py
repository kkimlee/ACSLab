import os
import csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models
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

C1_category = search('./weight_0', '')

C1_all = list()
C1_sp = list()
C1_un = list()
for data_dir in C1_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['fft']
        C1_all.append(data)
        
    C1_sp.append(pd.read_csv(C1_data[0])['fft'])
    for file in C1_data[1:]:
        C1_un.append(pd.read_csv(C1_data[0])['fft'])



C1_all = np.array(C1_all)[:, :]
C1_sp = np.array(C1_sp)[:, :]
C1_un = np.array(C1_un)[:, :]

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

history = autoencoder.fit(C1_all, C1_all,
       batch_size=64,
       epochs=40,
       validation_data=(C1_all, C1_all))

def encoder_loss(ae_loss, inter_loss):
    def loss(y_true, y_pred):
        cluster_loss = K.mean(K.mean((y_true - y_pred)) + abs(ae_loss) - abs(inter_loss))

        return cluster_loss
    return loss
    

for epi in range(50):   
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
    inter_loss = inter_loss/9
    
    sp_centroid = np.array(sp_centroid)
    sp_centorid = np.array(sp_centroid)
    
    for epoch in range(200):
        print('autoencoder')
        ae_loss = autoencoder.fit(C1_all, C1_all).history['loss'][0]
        
        print('encoder')
        encoder.compile(optimizer=Adam(lr=0.0001),
                        loss=encoder_loss(ae_loss=ae_loss, inter_loss=inter_loss))

        encoder.fit(C1_un, sp_centroid,
                    batch_size=64,
                    epochs=1)



    
    
    

    
'''
class AutoEncoder(Model):
    
    def __init__(self, encoder, decoder):
        
        super(AutoEncoder, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def call(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded

autoencoder = AutoEncoder(encoder, decoder)
autoencoder.summary()
'''
'''
Stage 1.
'''
'''
class AutoEncoder(Model):
    def __init__(self):
      
        super(AutoEncoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            # Conv1D(filters=32, kernel_size=10, activation=LeakyReLU(alpha=1), input_shape=(512, 1)),
            # Dropout(0.5),
            # Conv1D(filters=16, kernel_size=10,activation=LeakyReLU(alpha=1)),
            # Dropout(0.5),
            # Conv1D(filters=8, kernel_size=10, activation=LeakyReLU(alpha=1)),
            # Dropout(0.5),
            # Flatten(),
            # Dense(128)
            Flatten(),
            Dense(320, activation=LeakyReLU(alpha=1)),
            Dense(160, activation=LeakyReLU(alpha=1)),
            Dense(80, activation=LeakyReLU(alpha=1)),
            Dense(128)
            ])
        
        self.decoder = tf.keras.Sequential([
            # Reshape((128, 1)),
            # Conv1D(filters=32, kernel_size=10, activation=LeakyReLU(alpha=1)),
            # Dropout(0.5),
            # Conv1D(filters=16, kernel_size=10, activation=LeakyReLU(alpha=1)),
            # Dropout(0.5),
            # Conv1D(filters=8, kernel_size=10, activation=LeakyReLU(alpha=1)),
            # Dropout(0.5),
            # Flatten(),
            # Dense(512),
            # Reshape((512, 1))
            Flatten(),
            Dense(320, activation=LeakyReLU(alpha=1)),
            Dense(160, activation=LeakyReLU(alpha=1)),
            Dense(80, activation=LeakyReLU(alpha=1)),
            Dense(512)
            ])
        
      
    def latent(self, x):
        latent = self.encoder(x)
        
        return latent
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded
    
  



C1 = AutoEncoder()

C1.compile(optimizer=Adam(lr=0.0001),
              loss='mse')

C1.fit(C1_all, C1_all,
       batch_size=64,
       epochs=40,
       validation_data=(C1_all, C1_all))

'''
'''
Stage 2.
'''
'''
def centroid(AutoEncoder, x):
    
    latent = AutoEncoder.encoder(x)
        
    centroid = list()
    for i in range(len(latent[0])):
        sum_latent = list()
        for j in range(len(latent)):
            sum_latent.append(latent[j, i])
        centroid.append(np.mean(sum_latent))
    centroid = np.array(centroid).reshape(1, 128)
      
    return centroid

# Centroid
sp_centorid = centroid(C1, C1_sp)

# Clustered
un_latent = C1.latent(C1_un)
kmeans = KMeans(n_clusters=10).fit(un_latent)
un_centroid = kmeans.cluster_centers_
result = kmeans.predict(un_latent)
'''
'''
AutoEncoder의 
Encoder는 AutoEncoder 전체 loss와 Cluset 잘되게 끔 하는 Loss 모두 이용하여 학습
Decoder는 AutoEncoder의 Loss 만 학습
'''

'''
data = C1_train[0:10]
# data = np.reshape(data, data.shape+(1,))

result = C1.predict(data)
latent = C1.latent(C1_train)

centroid = list()
for i in range(len(C1_train)):
    centroid.append(np.mean(latent[i]))
    
plt.scatter(x = np.arange(0, len(centroid)), y = centroid)
plt.show()


for i in range(10):
    plt.plot(C1_train[i])
    plt.plot(result[i])
    plt.show()
'''