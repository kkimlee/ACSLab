# -*- coding: utf-8 -*-
"""
Created on Thu May 13 05:42:58 2021

@author: user
"""

import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import *
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from scipy import stats
from sklearn.preprocessing import LabelEncoder


import numpy as np
import math

raw_data = pd.read_excel('의료 데이터.xlsx')
one_hot_data = raw_data.fillna(0)


one_hot_age = one_hot_data['age 60']
for i in range(len(one_hot_age)):
    age = one_hot_age.iloc[i]
    if age == 0:
        one_hot_age[i] = 0
    elif age == 1:
        one_hot_age[i] = 1

one_hot_MLH1 = one_hot_data['MLH1']
for i in range(len(one_hot_MLH1)):
    MLH = str(one_hot_MLH1.iloc[i])
    if MLH == 'Intact' or MLH == 'intact':
        one_hot_MLH1.iloc[i] = 'intact'
    elif MLH == 'loss' or MLH == 'Partial loss':
        one_hot_MLH1.iloc[i] = 'loss'
    else:
        one_hot_MLH1.iloc[i] = 'none'
        
one_hot_MSH2 = one_hot_data['MSH2']
for i in range(len(one_hot_MSH2)):
    msh2 = str(one_hot_MSH2.iloc[i])
    if msh2 == 'intact' or msh2 == 'Intact':
        one_hot_MSH2.iloc[i] = 'intact'
    elif msh2 == 'loss' or msh2 == 'Partial loss':
        one_hot_MSH2.iloc[i] = 'loss'
    else:
        one_hot_MSH2.iloc[i] = 'none'

one_hot_MSH6 = one_hot_data['MSH6']
for i in range(len(one_hot_MSH6)):
    msh6 = str(one_hot_MSH6.iloc[i])
    if msh6 == 'intact' or msh6 == 'Intact':
        one_hot_MSH6.iloc[i] = 'intact'
    elif msh6 == 'loss' or msh6 == 'Partial loss' or msh6 == 'Loss':
        one_hot_MSH6.iloc[i] = 'loss'
    else:
        one_hot_MSH6.iloc[i] = 'none'
        
one_hot_PMS2 = one_hot_data['PMS2']
for i in range(len(one_hot_PMS2)):
    pms2 = str(one_hot_PMS2.iloc[i])
    if pms2 == 'intact' or pms2 == 'Intact' or pms2 == 'Intra': # 정상
        one_hot_PMS2.iloc[i] = 'intact'
    elif pms2 == 'loss' or pms2 == 'Partial loss': # 비정상
        one_hot_PMS2.iloc[i] = 'loss'
    else:
        one_hot_PMS2.iloc[i] = 'none'
        
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

one_hot_data = one_hot_data.drop(columns=['tumor size(mm)', '3M', '6M', '9M', 'result', '재발시점', '사망여부'])
one_hot_data = pd.get_dummies(one_hot_data)

sns.heatmap(one_hot_data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2, 
                vmax=1, vmin=-1, fmt='1.2f')
fig=plt.gcf()
fig.set_size_inches(12,9)
plt.title('one hot Heat map')
plt.show()

label = 'Relapse'

count = list()
for column in list(one_hot_data.columns):
    if column == label:
        continue
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


data_columns = list(one_hot_data.columns)
data_columns.remove(label)
for i in range(len(data_ex)):
    df = data_ex[i]
    df_bin = (df > 5)
    data_bin = list(df_bin.to_numpy().flatten())
    ratio = data_bin.count(False) / len(data_bin)
    print(data_columns[i])
    if ratio >= 0.2:
        data = count[i].drop(['total'])
        data = data.drop(columns=['total'])
        
        oddratio, pvalue = stats.fisher_exact(data)
        print('fisher oddratio : ', oddratio, ' p-value : ', pvalue)
    else:
        data = count[i].drop(['total'])
        data = data.drop(columns=['total'])
        
        chi_value = stats.chi2_contingency(data, correction=False)[0]
        p_value = stats.chi2_contingency(data, correction=False)[1]
        
        print('chi square value : ', chi_value, ", p-value : ", p_value)

train_data = one_hot_data.drop(columns=[label])
train_data_label = one_hot_data[label]

RF_model = RandomForestClassifier()
RF_model.fit(train_data, train_data_label)
result_RF_model = RF_model.predict(train_data)
RF_model_cm = confusion_matrix(train_data_label, result_RF_model)
print('RF result')
print(RF_model_cm)
print(classification_report(train_data_label, result_RF_model))


GradientBoosting_model = GradientBoostingClassifier()
GradientBoosting_model.fit(train_data, train_data_label)
result_GradientBoosting_model = GradientBoosting_model.predict(train_data)
GradientBoosting_model_cm = confusion_matrix(train_data_label, result_GradientBoosting_model)
print('GradientBoosting result')
print(GradientBoosting_model_cm)
print(classification_report(train_data_label, result_GradientBoosting_model))

LogisticRegression_model = LogisticRegression()
LogisticRegression_model.fit(train_data, train_data_label)
result_LogisticRegression_model = LogisticRegression_model.predict(train_data)
LogisticRegression_model_cm = confusion_matrix(train_data_label, result_LogisticRegression_model)
print('LogisticRegression result')
print(LogisticRegression_model_cm)
print(classification_report(train_data_label, result_LogisticRegression_model))

Linear_SVC_model = LinearSVC()
Linear_SVC_model.fit(train_data, train_data_label)
result_Linear_SVC_model = Linear_SVC_model.predict(train_data)
Linear_SVC_model_cm = confusion_matrix(train_data_label, result_Linear_SVC_model)
print('Linear_SVC result')
print(Linear_SVC_model_cm)
print(classification_report(train_data_label, result_Linear_SVC_model))

NB_model = GaussianNB()
NB_model.fit(train_data, train_data_label)
result_NB_model = NB_model.predict(train_data)
NB_model_cm = confusion_matrix(train_data_label, result_NB_model)
print('NB result')
print(NB_model_cm)
print(classification_report(train_data_label, result_NB_model))

Kernel_SVC_model = SVC()
Kernel_SVC_model.fit(train_data, train_data_label)
result_Kernel_SVC_model = Kernel_SVC_model.predict(train_data)
Kernel_SVC_model_cm = confusion_matrix(train_data_label, result_Kernel_SVC_model)
print('Kernel_SVC result')
print(Kernel_SVC_model_cm)
print(classification_report(train_data_label, result_Kernel_SVC_model))

fpr1, tpr1, thresholds1 = roc_curve(train_data_label, RF_model.predict_proba(train_data)[:, 1])
auc1 = auc(fpr1, tpr1)
fpr2, tpr2, thresholds1 = roc_curve(train_data_label, GradientBoosting_model.predict_proba(train_data)[:, 1])
auc2 = auc(fpr2, tpr2)
fpr3, tpr3, thresholds1 = roc_curve(train_data_label, LogisticRegression_model.decision_function(train_data))
auc3 = auc(fpr3, tpr3)
fpr4, tpr4, thresholds1 = roc_curve(train_data_label, Linear_SVC_model.decision_function(train_data))
auc4 = auc(fpr4, tpr4)
fpr5, tpr5, thresholds1 = roc_curve(train_data_label, NB_model.predict_proba(train_data)[:, 1])
auc5 = auc(fpr5, tpr5)
fpr6, tpr6, thresholds1 = roc_curve(train_data_label, Kernel_SVC_model.decision_function(train_data))
auc6 = auc(fpr6, tpr6)

plt.plot(fpr1, tpr1, 'o-', ms=2, label="RandomForestClassifier(" + str(auc1)[:4] + ")")
plt.plot(fpr2, tpr2, 'o-', ms=2, label="GradientBoostingClassifier(" + str(auc2)[:4] + ")")
plt.plot(fpr3, tpr3, 'o-', ms=2, label="LogisticRegression(" + str(auc3)[:4] + ")")
plt.plot(fpr4, tpr4, 'o-', ms=2, label="LinearSVC(" + str(auc4)[:4] + ")")
plt.plot(fpr5, tpr5, 'o-', ms=2, label="GaussianNB(" + str(auc5)[:4] + ")")
plt.plot(fpr6, tpr6, 'o-', ms=2, label="KernelSVC(" + str(auc6)[:4] + ")")

plt.legend()
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('Fall-Out')
plt.ylabel('Recall')
plt.title('ROC curve')
plt.show()