import os
import csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Input, Flatten, concatenate, LeakyReLU, Dropout, Reshape

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

class Autoencoder(Model):
  def __init__(self):
      
      super(Autoencoder, self).__init__()

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
          Dense(512, activation=LeakyReLU(alpha=1)),
          Dense(256, activation=LeakyReLU(alpha=1)),
          Dense(128, activation=LeakyReLU(alpha=1)),
          Dense(64, activation=LeakyReLU(alpha=1)),
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
          Dense(64, activation=LeakyReLU(alpha=1)),
          Dense(128, activation=LeakyReLU(alpha=1)),
          Dense(256, activation=LeakyReLU(alpha=1)),
          Dense(512, activation=LeakyReLU(alpha=1)),
          Dense(512)
        ])

  def call(self, x):
    encoded = self.encoder(x)
    # print(encoded)
    # encoded = np.reshape(encoded, encoded.shape+(1,))
    decoded = self.decoder(encoded)
    return decoded

  def latent(self, x):
      encoded = self.encoder(x)
      
      return encoded
  
C1_category = search('./weight_0', '')

C1_train = list()
C1_sp = list()
C1_un = list()
for data_dir in C1_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['fft']
        C1_train.append(data)
        
    C1_sp.append(pd.read_csv(C1_data[0])['fft'])
    for file in C1_data[1:]:
        C1_un.append(pd.read_csv(C1_data[0])['fft'])



C1_train = np.array(C1_train)[:, :]
C1_sp = np.array(C1_sp)[:, :]
C1_un = np.array(C1_un)[:, :]

C1 = Autoencoder()

C1.compile(optimizer=Adam(lr=0.0001),
              loss='mean_absolute_error')

C1.fit(C1_train, C1_train,
       batch_size=64,
       epochs=40,
       validation_data=(C1_train, C1_train))


sp_latent = C1.latent(C1_sp)
sp_centroid = list()
for i in range(len(sp_latent)):
    sp_centroid.append(np.mean(sp_latent[i]))


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