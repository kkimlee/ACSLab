import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from tensorflow.keras import utils
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, fbeta_score
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle

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
            df = df.flatten()
            
            label.append(i)
            data.append(df)
    
    data = np.array(data)
    
    # label = label_encoder.fit_transform(np.array(label))
    # label = utils.to_categorical(label, 3)
    
    return data, label

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

motor_GradientBoosting_model = GradientBoostingClassifier(random_state=42, max_depth=1)
motor_GradientBoosting_model.fit(train_motor_data, train_motor_data_label)
result_motor_GradientBoosting_model = motor_GradientBoosting_model.predict(test_motor_data)
motor_GradientBoosting_model_cm = confusion_matrix(test_motor_data_label, result_motor_GradientBoosting_model, labels=[0, 1, 2])
print('motor_GradientBoosting_model_cm')
print(motor_GradientBoosting_model_cm)

joblib.dump(motor_GradientBoosting_model, 'motor_GB.pkl') 

cover_GradientBoosting_model = GradientBoostingClassifier(random_state=42, max_depth=1)
cover_GradientBoosting_model.fit(train_cover_data, train_cover_data_label)
result_cover_GradientBoosting_model = cover_GradientBoosting_model.predict(test_cover_data)
cover_GradientBoosting_model_cm = confusion_matrix(test_cover_data_label, result_cover_GradientBoosting_model, labels=[0, 1, 2])
print('cover_GradientBoosting_model_cm')
print(cover_GradientBoosting_model_cm)

joblib.dump(cover_GradientBoosting_model, 'cover_GB.pkl') 

motor_LinearSVC_model = LinearSVC()
motor_LinearSVC_model.fit(train_motor_data, train_motor_data_label)
result_motor_LinearSVC_model = motor_LinearSVC_model.predict(test_motor_data)
motor_LinearSVC_model_cm = confusion_matrix(test_motor_data_label, result_motor_LinearSVC_model, labels=[0, 1, 2])
print('motor_LinearSVC_model_cm')
print(motor_LinearSVC_model_cm)

joblib.dump(motor_LinearSVC_model, 'motor_LinearSVC.pkl') 

cover_LinearSVC_model = LinearSVC()
cover_LinearSVC_model.fit(train_cover_data, train_cover_data_label)
result_cover_LinearSVC_model = cover_LinearSVC_model.predict(test_cover_data)
cover_LinearSVC_model_cm = confusion_matrix(test_cover_data_label, result_cover_LinearSVC_model, labels=[0, 1, 2])
print('cover_LinearSVC_model_cm')
print(cover_LinearSVC_model_cm)

joblib.dump(cover_LinearSVC_model, 'cover_LinearSVC.pkl') 

motor_LR_model = LogisticRegression(random_state=1)
motor_LR_model.fit(train_motor_data, train_motor_data_label)
result_motor_LR_model = motor_LR_model.predict(test_motor_data)
motor_LR_model_cm = confusion_matrix(test_motor_data_label, result_motor_LR_model, labels=[0, 1, 2])
print('motor_LR_model_cm')
print(motor_LR_model_cm)

joblib.dump(motor_LR_model, 'motor_LR.pkl') 

cover_LR_model = LogisticRegression(random_state=1)
cover_LR_model.fit(train_cover_data, train_cover_data_label)
result_cover_LR_model = cover_LR_model.predict(test_cover_data)
cover_LR_model_cm = confusion_matrix(test_cover_data_label, result_cover_LR_model, labels=[0, 1, 2])
print('cover_LR_model_cm')
print(cover_LR_model_cm)

joblib.dump(cover_LR_model, 'cover_LR.pkl')

motor_RF_model = RandomForestClassifier(criterion = 'entropy', n_estimators=10, max_depth=5,random_state=1,n_jobs=-1)
motor_RF_model.fit(train_motor_data,train_motor_data_label)
result_motor_RF_model = motor_RF_model.predict(test_motor_data)
motor_RF_model_cm = confusion_matrix(test_motor_data_label, result_motor_RF_model, labels=[0, 1, 2])
print('motor_RF_model_cm')
print(motor_RF_model_cm)

joblib.dump(motor_RF_model, 'motor_RF.pkl')

cover_RF_model = RandomForestClassifier(criterion = 'entropy', n_estimators=10, max_depth=5,random_state=1,n_jobs=-1)
cover_RF_model.fit(train_cover_data,train_cover_data_label)
result_cover_RF_model = cover_RF_model.predict(test_cover_data)
cover_RF_model_cm = confusion_matrix(test_cover_data_label, result_cover_RF_model, labels=[0, 1, 2])
print('cover_RF_model_cm')
print(cover_RF_model_cm)

joblib.dump(cover_RF_model, 'cover_RF.pkl')

