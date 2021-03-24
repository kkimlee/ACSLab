import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

label_encoder = LabelEncoder()

def search(dirname, extension):
    file_list = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == extension:
            file_list.append(full_filename)

    return file_list

train_images = search('./wt_train_opencv_img', '.png')
test_images = search('./wt_test_opencv_img', '.png')

x_train = []
x_test = []

y_train = []
y_test = []


for train_image in train_images:
    x_train.append(cv2.imread(train_image))
    
    if 'NO' in train_image:
        y_train.append('NO')
    else:
        y_train.append(train_image[11:20])

for test_image in test_images:
    x_test.append(cv2.imread(test_image))
    
    if 'NO' in test_image:
        y_test.append('NO')
    else:
        y_test.append(test_image[10:19])
        

x_train = np.array(x_train)[:, :, :, 0].astype(np.float32)
x_test = np.array(x_test)[:, :, :, 0].astype(np.float32)
x_train = np.reshape(x_train, x_train.shape+(1,))
x_test = np.reshape(x_test, x_test.shape+(1,))


y_train2 = y_train
y_test2 = y_test
y_train = label_encoder.fit_transform(np.array(y_train))
y_test = label_encoder.fit_transform(np.array(y_test))
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)


model = models.Sequential()
model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(12, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(240, activation='relu'))
model.add(layers.Dense(10, activation='sigmoid'))
model.summary()

'''
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
'''

model.compile(optimizer=Adam(lr=0.000002),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, batch_size=120, epochs=200)
cnn_pred = model.predict(x_test)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

extractor = keras.Model(inputs=model.inputs,
                       outputs=[layer.output for layer in model.layers])
train_feature_maps = extractor(x_train)
test_feature_maps = extractor(x_test)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_feature_maps = sess.run(train_feature_maps)
test_feature_maps = sess.run(test_feature_maps)

rf_1 = RandomForestClassifier()
rf_1.fit(np.reshape(train_feature_maps[1],(train_feature_maps[1].shape[0], -1)), y_train)
rf_1_pred = rf_1.predict(np.reshape(test_feature_maps[1], (test_feature_maps[1].shape[0], -1)))
rf_1_acc = accuracy_score(y_test, rf_1_pred)

rf_2 = RandomForestClassifier()
rf_2.fit(np.reshape(train_feature_maps[3],(train_feature_maps[3].shape[0], -1)), y_train)
rf_2_pred = rf_2.predict(np.reshape(test_feature_maps[3], (test_feature_maps[3].shape[0], -1)))
rf_2_acc = accuracy_score(y_test, rf_2_pred)

rf_3 = RandomForestClassifier()
rf_3.fit(np.reshape(train_feature_maps[5],(train_feature_maps[5].shape[0], -1)), y_train)
rf_3_pred = rf_3.predict(np.reshape(test_feature_maps[5], (test_feature_maps[5].shape[0], -1)))
rf_3_acc = accuracy_score(y_test, rf_3_pred)

print('cnn_acc :', test_acc)
print('rf_1_acc :', rf_1_acc)
print('rf_2_acc :', rf_2_acc)
print('rf_3_acc :', rf_3_acc)
