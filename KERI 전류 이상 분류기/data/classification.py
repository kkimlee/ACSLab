import os
import math
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling1D, MaxPooling2D, Input, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD, Adamax, Adagrad
from keras import backend as K

def search(dirname, extension):
    file_list = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == extension:
            file_list.append(full_filename)

    return file_list

def random_batch_sample(data, batch):
    rand_n = np.random.randint(0, len(data))
    while(rand_n > len(data) - batch):
        rand_n = np.random.randint(0, len(data))    
    
    return data[rand_n:rand_n+batch]
    

def generate_sample_data(data):
    data_set = []
    
    for i in range(400):
        sample_data = random_batch_sample(data, 10000)
        data_set.append(sample_data)
        
    return data_set
            
sess = tf.Session()
K.set_session(sess)

data_file = search('./', '.csv')

data_list = []
for file in data_file:
    df = pd.read_csv(file)
    data_list.append(df)

normal_data_set = []
close_data_set = []
abnormal_data_set = []

data = data_list[1]
data_columns = ['Va', 'Vb', 'Vc']

for i in range(len(data_columns)):
    fig = plt.figure()
    
    normal_data = data[data_columns[i]][:345000]
    close_data = data[data_columns[i]][345000:523000]
    abnormal_data = data[data_columns[i]][523000:615000]
    
    normal_data_set.append(generate_sample_data(normal_data))
    close_data_set.append(generate_sample_data(close_data))
    abnormal_data_set.append(generate_sample_data(abnormal_data))

normal_spectrogram_data_set = []
close_spectrogram_data_set = []
abnormal_spectrogram_data_set = []

for i in range(len(normal_data_set)):
    normal_spectrogram = []
    close_spectrogram = []
    abnormal_spectrogram = []
    for j in range(len(normal_data_set[i])):
        
        
        normal = np.array(normal_data_set[i][j])
        normal_D = np.array(np.abs(librosa.stft(normal)))
        #normal_D = np.where(normal_D > 0, normal_D, 0)
        normal_spectrogram.append(normal_D)
        
        '''
        fig = plt.figure()
        librosa.display.specshow(librosa.amplitude_to_db(normal_D, ref=np.max), y_axis='log', x_axis='time')
        plt.colorbar()
        plt.show()
        '''
        
        
        close = np.array(close_data_set[i][j])
        close_D = np.array(np.abs(librosa.stft(close)))
        #close_D = np.where(close_D > 0, close_D, 0)
        close_spectrogram.append(close_D)
        
        '''
        fig = plt.figure()
        librosa.display.specshow(librosa.amplitude_to_db(close_D, ref=np.max), y_axis='linear', x_axis='time')
        plt.colorbar()
        plt.show()
        '''
        
      
        
        abnormal = np.array(abnormal_data_set[i][j])
        abnormal_D = np.array(np.abs(librosa.stft(abnormal)))
        #abnormal_D = np.where(abnormal_D > 0, abnormal_D, 0)
        abnormal_spectrogram.append(abnormal_D)
        
        '''
        fig = plt.figure()
        librosa.display.specshow(librosa.amplitude_to_db(abnormal_D, ref=np.max), y_axis='linear', x_axis='time')
        plt.colorbar()
        plt.show()
        '''
        
    normal_spectrogram_data_set.append(normal_spectrogram)
    close_spectrogram_data_set.append(close_spectrogram)
    abnormal_spectrogram_data_set.append(abnormal_spectrogram)

train_data = []
train_label = []
for i in range(len(normal_data_set)):
    train_data.append(normal_spectrogram_data_set[i][:320]+close_spectrogram_data_set[i][:320]+abnormal_spectrogram_data_set[i][:320])
    
    label = []
    for j in range(len(train_data[i])):
        if j // 320 == 0:
            label.append(0)
        elif j // 320 == 1:
            label.append(1)
        else:
            label.append(2)
    train_label.append(label)


for i in range(len(train_data)):
    train_data[i] = np.array(train_data[i])    
    train_data[i] = np.reshape(train_data[i], train_data[i].shape + (1,))
    train_label[i] = np.array(train_label[i])
    train_label[i] = np.reshape(train_label[i], train_label[i].shape + (1,))

    

test_data = []
test_label = []
for i in range(len(normal_data_set)):
    test_data.append(normal_spectrogram_data_set[i][320:]+close_spectrogram_data_set[i][320:]+abnormal_spectrogram_data_set[i][320:])
    
    label = []
    for j in range(len(test_data[i])):
        if j // 80 == 0:
            label.append(0)
        elif j // 80 == 1:
            label.append(1)
        else:
            label.append(2)
    test_label.append(label)

for i in range(len(test_data)):
    test_data[i] = np.array(test_data[i])
    test_data[i] = np.reshape(test_data[i], test_data[i].shape + (1,))
    test_label[i] = np.array(test_label[i])
    test_label[i] = np.reshape(test_label[i], test_label[i].shape + (1,))

data_input_shape = train_data[0][0].shape

data_input = Input(shape=data_input_shape)
data_stack = Conv2D(filters=4, kernel_size=(128, 2), name="convolution0", padding='valid', activation='relu')(data_input)
data_stack = Conv2D(filters=4, kernel_size=(128, 2), name="convolution1", padding='valid', activation='relu')(data_stack)
data_stack = Conv2D(filters=8, kernel_size=(64, 2), name="convolution2", padding='valid', activation='relu')(data_stack)
data_stack = Conv2D(filters=8, kernel_size=(64, 2), name="convolution3", padding='valid', activation='relu')(data_stack)
data_stack = Conv2D(filters=16, kernel_size=(32, 1), name="convolution4", padding='valid', activation='relu')(data_stack)
data_stack = Conv2D(filters=16, kernel_size=(32, 1), name="convolution5", padding='valid', activation='relu')(data_stack)

data_stack = Flatten()(data_stack)

data_stack = Dense(60, activation='relu', name="dense0")(data_stack)
data_stack = Dropout(0.2)(data_stack)
data_stack = Dense(90, activation='relu', name="dense1")(data_stack)
data_stack = Dropout(0.2)(data_stack)
data_stack = Dense(120, activation='relu', name="dense2")(data_stack)

data_stack = Dense(3, activation='softmax', name="output")(data_stack)
adam = Adam(lr=0.000001)

pred = []
Va, Vb, Vc = [], [], []
result = [Va, Vb, Vc]
for i in range(len(test_data)):
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    model = Model(inputs=data_input, outputs=data_stack)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    history = model.fit(x=train_data[i], y=train_label[i], batch_size=64, epochs=500, validation_data=(test_data[i], test_label[i]))
    model.save('model_' + data_columns[i])
    pred.append(model.predict(test_data[i]))
    test_loss, test_acc = model.evaluate(test_data[i], test_label[i], verbose=2)
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    
    plt.figure(figsize=(8, 8)) 
    
    plt.subplot(2, 1, 1) 
    plt.plot(acc, label='Training Accuracy') 
    plt.plot(val_acc, label='Validation Accuracy') 
    plt.legend(loc='lower right') 
    plt.ylabel('Accuracy') 
    plt.ylim([min(plt.ylim()),1]) 
    plt.title('Training and Validation Accuracy') 
    
    plt.subplot(2, 1, 2) 
    plt.plot(loss, label='Training Loss') 
    plt.plot(val_loss, label='Validation Loss') 
    plt.legend(loc='upper right') 
    plt.ylabel('Cross Entropy') 
    plt.ylim([0,1.0]) 
    plt.title('Training and Validation Loss') 
    plt.xlabel('epoch') 
    
    plt.tight_layout()
    plt.show()

    
    D = []
    for j in range(len(data[data_columns[i]])//10000):
        tmp_data = data[data_columns[i]][j*10000:(j+1)*10000]
        tmp_data = np.array(tmp_data)
        D.append(np.abs(librosa.stft(tmp_data)))
    
    D = np.array(D)
    D = np.reshape(D, D.shape + (1,))
    result[i].append(model.predict(D))
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data[data_columns[i]])
    for x in range(len(data[data_columns[i]])//10000):
        ax.axvline(x=x*10000, ymin=0.0, ymax=1.0, color = 'black')
    plt.show()
        