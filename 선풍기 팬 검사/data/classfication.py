import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Input, Flatten, concatenate

def search(dirname, extension):
    file_list = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == extension:
            file_list.append(full_filename)

    return file_list

file_list = search('./', '')

motor_normal_data = []
motor_abnormal_data = []
cover_normal_data = []
cover_abnormal_data = []

for file in file_list:
    if 'motor' in file:
        if 'abnormal' in file:
            motor_abnormal_data_list = search(file + '/', '.csv')
        elif 'normal' in file:
            motor_normal_data_list = search(file + '/', '.csv')
    elif 'cover' in file:
        if 'abnormal' in file:
            cover_abnormal_data_list = search(file + '/', '.csv')
        elif 'normal' in file:
            cover_normal_data_list = search(file + '/', '.csv')

train_motor_data = []
train_motor_data_label = []
for data_file in motor_abnormal_data_list[:100]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    
    train_motor_data_label.append(1)
    train_motor_data.append(df)
    
for data_file in motor_normal_data_list[:100]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    
    train_motor_data_label.append(0)
    train_motor_data.append(df)
    
train_motor_data = np.array(train_motor_data)[:, :, :]
train_motor_data = np.reshape(train_motor_data, train_motor_data.shape + (1, ))

test_motor_data = []
test_motor_data_label = []
for data_file in motor_abnormal_data_list[100:]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    
    test_motor_data_label.append(1)
    test_motor_data.append(df)
    
for data_file in motor_normal_data_list[100:]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    
    test_motor_data_label.append(0)
    test_motor_data.append(df)

test_motor_data = np.array(test_motor_data)[:, :, :]
test_motor_data = np.reshape(test_motor_data, test_motor_data.shape + (1, ))

train_cover_data = []
train_cover_data_label = []
for data_file in cover_abnormal_data_list[:100]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    
    train_cover_data_label.append(1)
    train_cover_data.append(df)
    
for data_file in cover_normal_data_list[:100]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    
    train_cover_data_label.append(0)
    train_cover_data.append(df)
    
train_cover_data = np.array(train_cover_data)[:, :, :]
train_cover_data = np.reshape(train_cover_data, train_cover_data.shape + (1, ))

test_cover_data = []
test_cover_data_label = []
for data_file in cover_abnormal_data_list[100:]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    
    test_cover_data_label.append(1)
    test_cover_data.append(df)
    
for data_file in cover_normal_data_list[100:]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    
    test_cover_data_label.append(0)
    test_cover_data.append(df)

test_cover_data = np.array(test_cover_data)[:, :, :]
test_cover_data = np.reshape(test_cover_data, test_cover_data.shape + (1, ))

model = Sequential()
model.add(Conv2D(6, (3, 3), activation='relu', padding='same', input_shape=(1000, 9, 1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(12, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(240, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=Adam(lr=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x=train_motor_data, y=train_motor_data_label, batch_size=10, epochs=100)
cnn_pred = model.predict(test_motor_data)
test_loss, test_acc = model.evaluate(test_motor_data, test_motor_data_label, verbose=1)

model.fit(x=train_cover_data, y=train_cover_data_label, batch_size=10, epochs=100)
cnn_pred = model.predict(test_cover_data)
test_loss, test_acc = model.evaluate(test_cover_data, test_cover_data_label, verbose=1)