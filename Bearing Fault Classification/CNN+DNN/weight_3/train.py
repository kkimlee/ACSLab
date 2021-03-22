import os
import cv2
import csv
import numpy as np
import pandas as pd
from tensorflow.keras import utils
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Dropout, Flatten, Dense, Input, concatenate

def search(dirname, extension):
    file_list = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == extension:
            file_list.append(full_filename)

    return file_list

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

label_encoder = LabelEncoder()

train_label = train_data['label']
test_label = test_data['label']

train_label = label_encoder.fit_transform(np.array(train_label))
train_label = utils.to_categorical(train_label, 10)
test_label = label_encoder.fit_transform(np.array(test_label))
test_label = utils.to_categorical(test_label, 10)

train_feature = train_data.drop(['label'], axis=1)
test_feature = test_data.drop(['label'], axis=1)

train_value_feature = train_feature.drop(['img_path', 'raw_path'], axis=1)
test_value_feature = test_feature.drop(['img_path', 'raw_path'], axis=1)

train_img_feature = []
test_img_feature = []

train_raw_feature = []
test_raw_feature = []

for train_img_path in train_data['img_path']:
    train_img_feature.append(cv2.imread(train_img_path))
    
for test_img_path in test_data['img_path']:
    test_img_feature.append(cv2.imread(test_img_path))

for train_raw_path in train_data['raw_path']:
    train_raw_feature.append(pd.read_csv(train_raw_path)['Drive_End'])

for test_raw_path in test_data['raw_path']:
    test_raw_feature.append(pd.read_csv(test_raw_path)['Drive_End'])

train_value_feature = np.array(train_value_feature)
tset_value_feature = np.array(test_value_feature)
train_img_feature =  np.array(train_img_feature)[:, :, :, 0].astype(np.float32)
test_img_feature =  np.array(test_img_feature)[:, :, :, 0].astype(np.float32)
train_raw_feature = np.array(train_raw_feature)[:,:]
test_raw_feature = np.array(test_raw_feature)[:,:]

train_raw_feature = np.reshape(train_raw_feature, train_raw_feature.shape+(1,))
test_raw_feature = np.reshape(test_raw_feature, test_raw_feature.shape+(1,))


image_input_shape = train_img_feature[0].shape
value_input_shape = train_value_feature[0].shape
raw_input_shape = train_raw_feature[0].shape

raw_input = Input(shape=raw_input_shape)
raw_stack = Conv1D(filters=20, kernel_size=5, name="convolution0", padding='same', activation='relu')(raw_input)
raw_stack = MaxPooling1D(pool_size=2, name="maxpooling0")(raw_stack)
raw_stack = Conv1D(filters=40, kernel_size=5, name="convolution1", padding='same', activation='relu')(raw_input)
raw_stack = MaxPooling1D(pool_size=2, name="maxpooling1")(raw_stack)

raw_stack = Flatten()(raw_stack)

value_input = Input(shape=value_input_shape)
value_stack = Dense(20, activation='relu', name="dense0")(value_input)
value_stack = Dense(40, activation='relu', name="dense1")(value_stack)
value_stack = Dense(80, activation='relu', name="dense2")(value_stack)
value_stack = Dense(160, activation='relu', name="dense3")(value_stack)

merged = concatenate([raw_stack, value_stack])
merged = Dropout(0.5)(merged)
merged = Dense(10, activation='softmax', name="output")(merged)

adam = Adam(lr=0.0015)
model = Model(inputs=[raw_input, value_input], outputs=merged)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(x=[train_raw_feature, train_value_feature], y=train_label, batch_size=100, epochs=100)
cnn_pred = model.predict([test_raw_feature, test_value_feature])
test_loss, test_acc = model.evaluate([test_raw_feature, test_value_feature], test_label, verbose=2)
print(test_acc)