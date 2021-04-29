# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:19:54 2021

@author: user
"""
import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, fbeta_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def search(dirname, extension):
    file_list = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == extension:
            file_list.append(full_filename)
        
    return file_list

raw_data = pd.read_excel('의료 데이터.xlsx')
data = raw_data.fillna(0)

AGE = data['age 60']
for i in range(len(AGE)):
    age = AGE.iloc[i]
    if age == 0:
        AGE.iloc[i] = 1
    elif age == 1:
        AGE.iloc[i] = -1
    else:
        AGE.iloc[i] = 0

MLH1 = data['MLH1']
for i in range(len(MLH1)):
    MLH = str(MLH1.iloc[i])
    if MLH == 'Intact' or MLH == 'intact': # 정상
        MLH1.iloc[i] = 1
    elif MLH == 'loss' or MLH == 'Partial loss': # 비정상
        MLH1.iloc[i] = -1
    else:
        MLH1.iloc[i] = 0
data['MLH1'] = pd.to_numeric(data['MLH1'])

MSH2 = data['MSH2']
for i in range(len(MSH2)):
    msh2 = str(MSH2.iloc[i])
    if msh2 == 'intact' or msh2 == 'Intact': # 정상
        MSH2.iloc[i] = 1
    elif msh2 == 'loss' or msh2 == 'Partial loss': # 비정상
        MSH2.iloc[i] = -1
    else:
        MSH2.iloc[i] = 0
data['MSH2'] = pd.to_numeric(data['MSH2'])

MSH6 = data['MSH6']
for i in range(len(MSH6)):
    msh6 = str(MSH6.iloc[i])
    if msh6 == 'intact' or msh6 == 'Intact': # 정상
        MSH6.iloc[i] = 1
    elif msh6 == 'loss' or msh6 == 'Partial loss' or msh6 == 'Loss': # 비정상
        MSH6.iloc[i] = -1
    else:
        MSH6.iloc[i] = 0
data['MSH6'] = pd.to_numeric(data['MSH6'])

PMS2 = data['PMS2']
for i in range(len(PMS2)):
    pms2 = str(PMS2.iloc[i])
    if pms2 == 'intact' or pms2 == 'Intact' or pms2 == 'Intra': # 정상
        PMS2.iloc[i] = 1
    elif pms2 == 'loss' or pms2 == 'Partial loss': # 비정상
        PMS2.iloc[i] = -1
    else:
        PMS2.iloc[i] = 0
data['PMS2'] = pd.to_numeric(data['PMS2'])

MSI = data['MSI']
for i in range(len(MSI)):
    msi = str(MSI.iloc[i])
    if msi[0:4] == 'high':
        MSI.iloc[i] = -2
    elif msi[0:3] == 'low':
        MSI.iloc[i] = -1
    elif msi == 'stable' or msi == 'Stable':
        MSI.iloc[i] = 1
    else:
        MSI.iloc[i] = 0
data['MSI'] = pd.to_numeric(data['MSI'])

_3M = data['3M']
for i in range(len(_3M)):
    _3m = str(_3M.iloc[i])
    if _3m[-2:] == 'SD':
        _3M.iloc[i] = -2
    elif _3m[-2:] == 'PR':
        _3M.iloc[i] = -1
    elif _3m[-2:] == 'PD':
        _3M.iloc[i] = -3
    elif _3m[-2:] == 'CR':
        _3M.iloc[i] = 1
    else:
        _3M.iloc[i] = 0
data['3M'] = pd.to_numeric(data['3M'])

_6M = data['6M']
for i in range(len(_6M)):
    _6m = str(_6M.iloc[i])
    if _6m[-2:] == 'SD':
        _6M.iloc[i] = _3M.iloc[i]
    elif _6m[-2:] == 'PR':
        _6M.iloc[i] = -1
    elif _6m[-2:] == 'PD':
        _6M.iloc[i] = -3
    elif _6m[-2:] == 'CR':
        _6M.iloc[i] = 1
    else:
        _6M.iloc[i] = 0
data['6M'] = pd.to_numeric(data['6M'])

_9M = data['9M']
for i in range(len(_9M)):
    _9m = str(_9M.iloc[i])
    if _9m[-2:] == 'SD':
        _9M.iloc[i] = _6M.iloc[i]
    elif _9m[-2:] == 'PR':
        _9M.iloc[i] = -1
    elif _9m[-2:] == 'PD':
        _9M.iloc[i] = -3
    elif _9m[-2:] == 'CR':
        _9M.iloc[i] = 1
    else:
        _9M.iloc[i] = 0
data['9M'] = pd.to_numeric(data['9M'])

Result = data['result']
for i in range(len(_9M)):
    result = str(Result.iloc[i])
    if result == 'SD':
        Result.iloc[i] = _9M.iloc[i]
    elif result == 'PR':
        Result.iloc[i] = -1
    elif result == 'PD':
        Result.iloc[i] = -3
    elif result == 'CR':
        Result.iloc[i] = 1   
    else:
        Result.iloc[i] = 0
data['result'] = pd.to_numeric(data['result'])

Relapse = data['Relapse']
for i in range(len(Relapse)):
    relapse = Relapse.iloc[i]
    if relapse == 1:
        Relapse[i] = -1
    elif relapse == 0:
        Relapse[i] = 1
    else:
        Relapse[i] = 0
data['Relapse'] = pd.to_numeric(data['Relapse'])

data = data.drop(columns='재발시점')
data = data.drop(columns='사망여부')

sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2, 
                vmax=1, vmin=-1, fmt='1.2f')
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.title('Heat map')
plt.show()

train_data = data.drop(columns=['3M', '6M', '9M', 'result', 'Relapse'])
train_data_label = data['Relapse']

LR_model = LogisticRegression()
LR_model.fit(train_data, train_data_label)
result_LR_model = LR_model.predict(train_data)
LR_model_cm = confusion_matrix(train_data_label, result_LR_model)
print(LR_model_cm)

LinearSVC_model = LinearSVC()
LinearSVC_model.fit(train_data, train_data_label)
result_LinearSVC_model = LinearSVC_model.predict(train_data)
LinearSVC_model_cm = confusion_matrix(train_data_label, result_LinearSVC_model)
print(LinearSVC_model_cm)

RF_model = RandomForestClassifier()
RF_model.fit(train_data, train_data_label)
result_RF_model = RF_model.predict(train_data)
RF_model_cm = confusion_matrix(train_data_label, result_RF_model)
print(RF_model_cm)

GradientBoosting_model = GradientBoostingClassifier()
GradientBoosting_model.fit(train_data, train_data_label)
result_GradientBoosting_model = GradientBoosting_model.predict(train_data)
GradientBoosting_model_cm = confusion_matrix(train_data_label, result_GradientBoosting_model)
print(GradientBoosting_model_cm)