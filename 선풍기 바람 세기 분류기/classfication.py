import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Input, Flatten, concatenate

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

def search(dirname, extension):
    file_list = list()
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == extension:
            file_list.append(full_filename)

    return file_list

def search_data(path, category):
    train_file_list = search(path, '')

    data = list()
    for file in train_file_list:
        if category in file:
            if '1단계' in file:
                data_list = search(file + '/', '.csv')
                data.append(data_list)
            elif '2단계' in file:
                data_list = search(file + '/', '.csv')
                data.append(data_list)
            elif '3단계' in file:
                data_list = search(file + '/', '.csv')
                data.append(data_list)
    
    return data

def generate_data(data, label, data_file_list):
    for i in range(len(data_file_list)):
        for data_file in data_file_list[i]:
            df = pd.read_csv(data_file)
            df = df.drop('Record Time', axis=1)
            df = np.array(df)
            
            label.append(i)
            data.append(df)
    
    data = np.array(data)[:, :, :]
    data = np.reshape(data, data.shape + (1, ))
    
    label = label_encoder.fit_transform(np.array(label))
    label = utils.to_categorical(label, 3)
    
    return data, label

def create_model():
    model = Sequential()
    model.add(Conv2D(6, (3, 3), activation='relu', padding='same', input_shape=(10, 9, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(12, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(240, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    return model

train_data = search_data('./data/train', '모터')
train_motor_data = list()
train_motor_data_label = list()
train_motor_data, train_motor_data_label = generate_data(train_motor_data, train_motor_data_label, train_data)

test_data = search_data('./data/test', '모터')
test_motor_data = list()
test_motor_data_label = list()
test_motor_data, test_motor_data_label = generate_data(test_motor_data, test_motor_data_label, test_data)

train_data = search_data('./data/train', '커버')
train_cover_data = list()
train_cover_data_label = list()
train_cover_data, train_cover_data_label = generate_data(train_cover_data, train_cover_data_label, train_data)

test_data = search_data('./data/test', '커버')
test_cover_data = list()
test_cover_data_label = list()
test_cover_data, test_cover_data_label = generate_data(test_cover_data, test_cover_data_label, test_data)

motor_model = create_model()
motor_model.summary()
motor_model.compile(optimizer=Adam(lr=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

motor_model.fit(x=train_motor_data, y=train_motor_data_label, batch_size=10, epochs=100)
motor_model_pred = motor_model.predict(test_motor_data)
motor_model_test_loss, motor_model_test_acc = motor_model.evaluate(test_motor_data, test_motor_data_label, verbose=1)

motor_model.save('dl_motor_model.h5')

cover_model = create_model()
cover_model.summary()
cover_model.compile(optimizer=Adam(lr=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

cover_model.fit(x=train_cover_data, y=train_cover_data_label, batch_size=10, epochs=100)
cover_modelpred = cover_model.predict(test_cover_data)
cover_model_test_loss, cover_model_test_acc = cover_model.evaluate(test_cover_data, test_cover_data_label, verbose=1)

cover_model.save('dl_cover_model.h5')