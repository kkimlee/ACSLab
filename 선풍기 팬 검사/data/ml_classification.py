import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, fbeta_score


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
    df = df.flatten()
    
    train_motor_data_label.append(1)
    train_motor_data.append(df)
    
for data_file in motor_normal_data_list[:100]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    df = df.flatten()

    train_motor_data_label.append(0)
    train_motor_data.append(df)

train_motor_data = np.array(train_motor_data)
# train_motor_data = np.array(train_motor_data)[:, :, :]
# train_motor_data = np.reshape(train_motor_data, train_motor_data.shape + (1, ))

test_motor_data = []
test_motor_data_label = []
for data_file in motor_abnormal_data_list[100:]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    df = df.flatten()
    
    test_motor_data_label.append(1)
    test_motor_data.append(df)
    
for data_file in motor_normal_data_list[100:]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    df = df.flatten()
    
    test_motor_data_label.append(0)
    test_motor_data.append(df)

test_motor_data = np.array(test_motor_data)
# test_motor_data = np.array(test_motor_data)[:, :, :]
# test_motor_data = np.reshape(test_motor_data, test_motor_data.shape + (1, ))

train_cover_data = []
train_cover_data_label = []
for data_file in cover_abnormal_data_list[:100]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    df = df.flatten()
    
    train_cover_data_label.append(1)
    train_cover_data.append(df)
    
for data_file in cover_normal_data_list[:100]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    df = df.flatten()
    
    train_cover_data_label.append(0)
    train_cover_data.append(df)

train_cover_data = np.array(train_cover_data)
# train_cover_data = np.array(train_cover_data)[:, :, :]
# train_cover_data = np.reshape(train_cover_data, train_cover_data.shape + (1, ))

test_cover_data = []
test_cover_data_label = []
for data_file in cover_abnormal_data_list[100:]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    df = df.flatten()
    
    test_cover_data_label.append(1)
    test_cover_data.append(df)
    
for data_file in cover_normal_data_list[100:]:
    df = pd.read_csv(data_file)
    df = df.drop('Record Time', axis=1)
    df = np.array(df)
    df = df.flatten()
    
    test_cover_data_label.append(0)
    test_cover_data.append(df)
    
test_cover_data = np.array(test_cover_data)
# test_cover_data = np.array(test_cover_data)[:, :, :]
# test_cover_data = np.reshape(test_cover_data, test_cover_data.shape + (1, ))

motor_GradientBoosting_model = GradientBoostingClassifier(random_state=42, max_depth=1)
motor_GradientBoosting_model.fit(train_motor_data, train_motor_data_label)
result_motor_GradientBoosting_model = motor_GradientBoosting_model.predict(test_motor_data)
motor_GradientBoosting_model_cm = confusion_matrix(test_motor_data_label, result_motor_GradientBoosting_model, labels=[0, 1])
print('motor_GradientBoosting_model_cm')
print(motor_GradientBoosting_model_cm)

cover_GradientBoosting_model = GradientBoostingClassifier(random_state=42, max_depth=1)
cover_GradientBoosting_model.fit(train_cover_data, train_cover_data_label)
result_cover_GradientBoosting_model = cover_GradientBoosting_model.predict(test_cover_data)
cover_GradientBoosting_model_cm = confusion_matrix(test_cover_data_label, result_cover_GradientBoosting_model, labels=[0, 1])
print('cover_GradientBoosting_model_cm')
print(cover_GradientBoosting_model_cm)

motor_LinearSVC_model = LinearSVC()
motor_LinearSVC_model.fit(train_motor_data, train_motor_data_label)
result_motor_LinearSVC_model = motor_LinearSVC_model.predict(test_motor_data)
motor_LinearSVC_model_cm = confusion_matrix(test_motor_data_label, result_motor_LinearSVC_model, labels=[0, 1])
print('motor_LinearSVC_model_cm')
print(motor_LinearSVC_model_cm)

cover_LinearSVC_model = LinearSVC()
cover_LinearSVC_model.fit(train_cover_data, train_cover_data_label)
result_cover_LinearSVC_model = cover_LinearSVC_model.predict(test_cover_data)
cover_LinearSVC_model_cm = confusion_matrix(test_cover_data_label, result_cover_LinearSVC_model, labels=[0, 1])
print('cover_LinearSVC_model_cm')
print(cover_LinearSVC_model_cm)

motor_LR_model = LogisticRegression(random_state=1)
motor_LR_model.fit(train_motor_data, train_motor_data_label)
result_motor_LR_model = motor_LR_model.predict(test_motor_data)
motor_LR_model_cm = confusion_matrix(test_motor_data_label, result_motor_LR_model, labels=[0, 1])
print('motor_LR_model_cm')
print(motor_LR_model_cm)

cover_LR_model = LogisticRegression(random_state=1)
cover_LR_model.fit(train_cover_data, train_cover_data_label)
result_cover_LR_model = cover_LR_model.predict(test_cover_data)
cover_LR_model_cm = confusion_matrix(test_cover_data_label, result_cover_LR_model, labels=[0, 1])
print('cover_LR_model_cm')
print(cover_LR_model_cm)

motor_RF_model = RandomForestClassifier(criterion = 'entropy', n_estimators=10, max_depth=5,random_state=1,n_jobs=-1)
motor_RF_model.fit(train_motor_data,train_motor_data_label)
result_motor_RF_model = motor_RF_model.predict(test_motor_data)
motor_RF_model_cm = confusion_matrix(test_motor_data_label, result_motor_RF_model, labels=[0, 1])
print('motor_RF_model_cm')
print(motor_RF_model_cm)

cover_RF_model = RandomForestClassifier(criterion = 'entropy', n_estimators=10, max_depth=5,random_state=1,n_jobs=-1)
cover_RF_model.fit(train_cover_data,train_cover_data_label)
result_cover_RF_model = cover_RF_model.predict(test_cover_data)
cover_RF_model_cm = confusion_matrix(test_cover_data_label, result_cover_RF_model, labels=[0, 1])
print('cover_RF_model_cm')
print(cover_RF_model_cm)


