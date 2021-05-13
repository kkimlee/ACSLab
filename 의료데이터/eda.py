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

from scipy import stats
from sklearn.preprocessing import LabelEncoder

import numpy as np
import math

def search(dirname, extension):
    file_list = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == extension:
            file_list.append(full_filename)
        
    return file_list

label_encoder = LabelEncoder()

raw_data = pd.read_excel('의료 데이터.xlsx')
data = raw_data.fillna(0)
one_hot_data = raw_data.fillna(0)

AGE = data['age 60']
for i in range(len(AGE)):
    age = AGE.iloc[i]
    if age == 0:
        AGE.iloc[i] = 1
    elif age == 1:
        AGE.iloc[i] = -1
    else:
        AGE.iloc[i] = 0

one_hot_age = one_hot_data['age 60']
for i in range(len(one_hot_age)):
    age = one_hot_age.iloc[i]
    if age == 0:
        one_hot_age[i] = 0
    elif age == 1:
        one_hot_age[i] = 1
# one_hot_data['age 60'] = label_encoder.fit_transform(np.array(one_hot_age))

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

one_hot_MLH1 = one_hot_data['MLH1']
for i in range(len(one_hot_MLH1)):
    MLH = str(one_hot_MLH1.iloc[i])
    if MLH == 'Intact' or MLH == 'intact':
        one_hot_MLH1.iloc[i] = 'intact'
    elif MLH == 'loss' or MLH == 'Partial loss':
        one_hot_MLH1.iloc[i] = 'loss'
    else:
        one_hot_MLH1.iloc[i] = 'none'
# one_hot_data['MLH1'] = label_encoder.fit_transform(np.array(one_hot_MLH1))

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

one_hot_MSH2 = one_hot_data['MSH2']
for i in range(len(one_hot_MSH2)):
    msh2 = str(one_hot_MSH2.iloc[i])
    if msh2 == 'intact' or msh2 == 'Intact':
        one_hot_MSH2.iloc[i] = 'intact'
    elif msh2 == 'loss' or msh2 == 'Partial loss':
        one_hot_MSH2.iloc[i] = 'loss'
    else:
        one_hot_MSH2.iloc[i] = 'none'
# one_hot_data['MSH2'] = label_encoder.transform(np.array(one_hot_MSH2))

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

one_hot_MSH6 = one_hot_data['MSH6']
for i in range(len(one_hot_MSH6)):
    msh6 = str(one_hot_MSH6.iloc[i])
    if msh6 == 'intact' or msh6 == 'Intact':
        one_hot_MSH6.iloc[i] = 'intact'
    elif msh6 == 'loss' or msh6 == 'Partial loss' or msh6 == 'Loss':
        one_hot_MSH6.iloc[i] = 'loss'
    else:
        one_hot_MSH6.iloc[i] = 'none'
# one_hot_data['MSH6'] = label_encoder.transform(np.array(one_hot_MSH6))

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

one_hot_PMS2 = one_hot_data['PMS2']
for i in range(len(one_hot_PMS2)):
    pms2 = str(one_hot_PMS2.iloc[i])
    if pms2 == 'intact' or pms2 == 'Intact' or pms2 == 'Intra': # 정상
        one_hot_PMS2.iloc[i] = 'intact'
    elif pms2 == 'loss' or pms2 == 'Partial loss': # 비정상
        one_hot_PMS2.iloc[i] = 'loss'
    else:
        one_hot_PMS2.iloc[i] = 'none'
# one_hot_data['PMS2'] = label_encoder.transform(np.array(one_hot_PMS2))

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

one_hot_MSI = one_hot_data['MSI']
for i in range(len(one_hot_MSI)):
    msi = str(one_hot_MSI.iloc[i])
    if msi[0:4] == 'high':
        one_hot_MSI.iloc[i] = 'high'
    elif msi[0:3] == 'low':
        one_hot_MSI.iloc[i] = 'low'
    elif msi == 'stable' or msi == 'Stable':
        one_hot_MSI.iloc[i] = 'stable'
    else:
        one_hot_MSI.iloc[i] = 'none'
# one_hot_data['MSI'] = label_encoder.fit_transform(np.array(one_hot_MSI))

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

one_hot_3m = one_hot_data['3M']
for i in range(len(one_hot_3m)):
    _3m = str(one_hot_3m.iloc[i])
    if _3m[-2:] == 'SD':
        one_hot_3m.iloc[i] = 'SD'
    elif _3m[-2:] == 'PR':
        one_hot_3m.iloc[i] = 'PR'
    elif _3m[-2:] == 'PD':
        one_hot_3m.iloc[i] = 'PD'
    elif _3m[-2:] == 'CR':
        one_hot_3m.iloc[i] = 'CR'
    else:
        one_hot_3m.iloc[i] = 'none'
# one_hot_data['3M'] = label_encoder.fit_transform(np.array(one_hot_3m))

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

one_hot_6m = one_hot_data['6M']
for i in range(len(one_hot_6m)):
    _6m = str(one_hot_6m.iloc[i])
    if _6m[-2:] == 'SD':
        one_hot_6m.iloc[i] = 'SD'
    elif _6m[-2:] == 'PR':
        one_hot_6m.iloc[i] = 'PR'
    elif _6m[-2:] == 'PD':
        one_hot_6m.iloc[i] = 'PD'
    elif _6m[-2:] == 'CR':
        one_hot_6m.iloc[i] = 'CR'
    else:
        one_hot_6m.iloc[i] = 'none'
# one_hot_data['6M'] = label_encoder.transform(np.array(one_hot_6m))

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

one_hot_9m = one_hot_data['9M']
for i in range(len(one_hot_9m)):
    _9m = str(one_hot_9m.iloc[i])
    if _9m[-2:] == 'SD':
        one_hot_9m.iloc[i] = 'SD'
    elif _9m[-2:] == 'PR':
        one_hot_9m.iloc[i] = 'PR'
    elif _9m[-2:] == 'PD':
        one_hot_9m.iloc[i] = 'PD'
    elif _9m[-2:] == 'CR':
        one_hot_9m.iloc[i] = 'CR'
    else:
        one_hot_9m.iloc[i] = 'none'
# one_hot_data['9M'] = label_encoder.transform(np.array(one_hot_9m))

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

one_hot_result = one_hot_data['result']
for i in range(len(one_hot_9m)):
    result = str(one_hot_result.iloc[i])
    if result[-2:] == 'SD':
        one_hot_result.iloc[i] = 'SD'
    elif result[-2:] == 'PR':
        one_hot_result.iloc[i] = 'PR'
    elif result[-2:] == 'PD':
        one_hot_result.iloc[i] = 'PD'
    elif result[-2:] == 'CR':
        one_hot_result.iloc[i] = 'CR'
    else:
        one_hot_result.iloc[i] = 'none'
# one_hot_data['result'] = label_encoder.transform(np.array(one_hot_result))

Relapse = data['Relapse']
one_hot_data['Relapse'] = label_encoder.fit_transform(np.array(Relapse))
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

one_hot_data = one_hot_data.drop(columns='재발시점')
one_hot_data = one_hot_data.drop(columns='사망여부')

sns.heatmap(one_hot_data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2, 
                vmax=1, vmin=-1, fmt='1.2f')
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.title('one hot Heat map')
plt.show()


label = 'Relapse'

one_hot_data = one_hot_data.drop(columns=['tumor size(mm)'])

count = list()
for column in list(one_hot_data.columns[:-5]):
    data_bin_count = np.zeros((len(one_hot_data[column].unique()), len(one_hot_data[label].unique())))
    data_bin = list(one_hot_data.groupby([column, label]).size())

    for i in range(len(data_bin)):
        data_bin_count[i//len(data_bin_count[0])][i%len(data_bin_count[0])] = data_bin[i]
    index = list(one_hot_data[column].unique())
    index.sort()
    df = pd.DataFrame(data_bin_count, index=index)
    df['total'] = df[0] + df[1]
    df.loc['total'] = df.sum()

    count.append(df)

data_ex = list()
for i in range(len(count)):
    df = count[i]
    data_ratio_column = pd.DataFrame()
    data_ratio_row =pd.DataFrame()
    for i in range(len(df.columns[:-1])):
        column = df.columns[i]
        data_ratio_column[column] = (df[column] / df['total'])
        data_ratio_row[column] = (df[column]/df[column].loc['total'])
    data_ex.append((data_ratio_column * data_ratio_row * df['total'].loc['total']).drop(['total']))
    
train_data = data.drop(columns=['3M', '6M', '9M', 'result', 'Relapse'])
train_data_label = data[label]

one_hot_train_data = one_hot_data.drop(columns=['3M', '6M', '9M', 'result', 'Relapse'])
one_hot_train_data_label = one_hot_data[label]

'''
chi_square_result = list()
for column in train_data.columns:
    test_data = pd.DataFrame(train_data[column].astype('category'))[column]
    target_data = pd.DataFrame(train_data_label.astype('category'))[label]
    
    chi_data = pd.crosstab(test_data, target_data)
    chi_square_result.append(stats.chi2_contingency(chi_data, correction=False))
    
    chi_value = stats.chi2_contingency(chi_data, correction=False)[0]
    p_value = stats.chi2_contingency(chi_data, correction=False)[1]
    
    print(column, '과 ',label,'의 Chi Square : ', chi_value)
    print(column, '과 ',label,'의 p-value : ', p_value)
    
    df = stats.chi2_contingency(chi_data, correction=False)[2]
    # x = np.linspace(0, chi_value, 201)
    x = np.linspace(0, math.ceil(chi_value), 201)
    y = stats.chi2(df).pdf(x)
    
    x95 = stats.chi2(df).ppf(.95)
    
    plt.figure(figsize=(10,6))
    plt.plot(x, y, 'b--')
    
    plt.axvline(x=x95, color='black', linestyle=':')
    plt.text(x95,0, 'critical value\n' + str(round(x95, 4)),
             horizontalalignment='left', color='b')
    
    plt.axvline(x=chi_value, color='r', linestyle=':')
    plt.text(chi_value,0, 'statistic\n' + str(round(chi_value, 4)),
             horizontalalignment='left', color='b')
    
    plt.xlabel(label + '-' + column + ' X')
    plt.ylabel(label + '-' + column + ' P(X)')
    plt.grid()
    plt.title(label + '-' + column + r' $\chi^2$ Distribution (df = 2)')
    plt.show()
'''
'''
one_hot_chi_square_result = list()
for column in train_data.columns:
    one_hot_test_data = pd.DataFrame(one_hot_train_data[column].astype('category'))[column]
    one_hot_target_data = pd.DataFrame(one_hot_train_data_label.astype('category'))[label]
    
    one_hot_chi_data = pd.crosstab(one_hot_test_data, one_hot_target_data)
    one_hot_chi_square_result.append(stats.chi2_contingency(one_hot_chi_data, correction=False))
    
    chi_value = stats.chi2_contingency(one_hot_chi_data, correction=False)[0]
    p_value = stats.chi2_contingency(one_hot_chi_data, correction=False)[1]
    
    print(column, '과 ',label,'의 Chi Square : ', chi_value)
    print(column, '과 ',label,'의 p-value : ', p_value)
    
    df = stats.chi2_contingency(one_hot_chi_data, correction=False)[2]
    # x = np.linspace(0, chi_value, 201)
    x = np.linspace(0, math.ceil(chi_value), 201)
    y = stats.chi2(df).pdf(x)
    
    x95 = stats.chi2(df).ppf(.95)
    
    plt.figure(figsize=(10,6))
    plt.plot(x, y, 'b--')
    
    plt.axvline(x=x95, color='black', linestyle=':')
    plt.text(x95,0, 'critical value\n' + str(round(x95, 4)),
             horizontalalignment='left', color='b')
    
    plt.axvline(x=chi_value, color='r', linestyle=':')
    plt.text(chi_value,0, 'statistic\n' + str(round(chi_value, 4)),
             horizontalalignment='left', color='b')
    
    plt.xlabel(label + '-' + column + ' X')
    plt.ylabel(label + '-' + column + ' P(X)')
    plt.grid()
    plt.title(label + '-' + column + r' $\chi^2$ Distribution (df = 2)')
    plt.show()
'''
one_hot_fisher_result = list()
for column in train_data.columns:
    one_hot_test_data = pd.DataFrame(one_hot_train_data[column].astype('category'))[column]
    one_hot_target_data = pd.DataFrame(one_hot_train_data_label.astype('category'))[label]
    
    one_hot_fisher_data = pd.crosstab(one_hot_test_data, one_hot_target_data)
    one_hot_fisher_result.append(stats.fisher_exact(one_hot_fisher_data))

'''
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
'''