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

def search(dirname, extension):
    file_list = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == extension:
            file_list.append(full_filename)
        
    return file_list

data = pd.read_excel('의료 데이터.xlsx')
data = data.fillna(0)

MLH1 = data['MLH1']
for i in range(len(MLH1)):
    MLH = str(MLH1.iloc[i])
    if MLH == 'not' or MLH == '0': # 결측
        MLH1.iloc[i] = -4
    elif MLH == 'Intact' or MLH == 'intact': # 정상
        MLH1.iloc[i] = 1
    elif MLH == 'loss' or MLH == 'Partial loss': # 비정상
        MLH1.iloc[i] = -1
    else:
        MLH1.iloc[i] = -4
data['MLH1'] = pd.to_numeric(data['MLH1'])

MSH2 = data['MSH2']
for i in range(len(MSH2)):
    msh2 = str(MSH2.iloc[i])
    if msh2 == 'not' or msh2 == '0': # 결측
        MSH2.iloc[i] = -4
    elif msh2 == 'intact' or msh2 == 'Intact': # 정상
        MSH2.iloc[i] = 1
    elif msh2 == 'loss' or msh2 == 'Partial loss': # 비정상
        MSH2.iloc[i] = -1
    else:
        MSH2.iloc[i] = -4
data['MSH2'] = pd.to_numeric(data['MSH2'])

MSH6 = data['MSH6']
for i in range(len(MSH6)):
    msh6 = str(MSH6.iloc[i])
    if msh6 == 'not' or msh6 == '0': # 결측
        MSH6.iloc[i] = -4
    elif msh6 == 'intact' or msh6 == 'Intact': # 정상
        MSH6.iloc[i] = 1
    elif msh6 == 'loss' or msh6 == 'Partial loss' or msh6 == 'Loss': # 비정상
        MSH6.iloc[i] = -1
    else:
        MSH6.iloc[i] = -4
data['MSH6'] = pd.to_numeric(data['MSH6'])

PMS2 = data['PMS2']
for i in range(len(PMS2)):
    pms2 = str(PMS2.iloc[i])
    if pms2 == 'not' or pms2 == '0': # 결측
        PMS2.iloc[i] = -4
    elif pms2 == 'intact' or pms2 == 'Intact' or pms2 == 'Intra': # 정상
        PMS2.iloc[i] = 1
    elif pms2 == 'loss' or pms2 == 'Partial loss': # 비정상
        PMS2.iloc[i] = -1
    else:
        PMS2.iloc[i] = -4
data['PMS2'] = pd.to_numeric(data['PMS2'])

MSI = data['MSI']
for i in range(len(MSI)):
    msi = str(MSI.iloc[i])
    if msi == 'not' or msi == '0': # 결측
        MSI.iloc[i] = -4
    elif msi[0:4] == 'high':
        MSI.iloc[i] = -2
    elif msi[0:3] == 'low':
        MSI.iloc[i] = -1
    elif msi == 'stable' or msi == 'Stable':
        MSI.iloc[i] = 1
    else:
        MSI.iloc[i] = -4
data['MSI'] = pd.to_numeric(data['MSI'])

_3M = data['3M']
for i in range(len(_3M)):
    _3m = str(_3M.iloc[i])
    if _3m[-2:] == 'SD':
        _3M.iloc[i] = 0
    elif _3m[-2:] == 'PR':
        _3M.iloc[i] = 1
    elif _3m[-2:] == 'PD':
        _3M.iloc[i] = -1
    elif _3m[-2:] == 'CR':
        _3M.iloc[i] = 2
    else:
        _3M.iloc[i] = -4
data['3M'] = pd.to_numeric(data['3M'])

_6M = data['6M']
for i in range(len(_6M)):
    _6m = str(_6M.iloc[i])
    if _6m[-2:] == 'SD':
        _6M.iloc[i] = -2
    elif _6m[-2:] == 'PR':
        _6M.iloc[i] = -1
    elif _6m[-2:] == 'PD':
        _6M.iloc[i] = -3
    elif _6m[-2:] == 'CR':
        _6M.iloc[i] = 1
    else:
        _6M.iloc[i] = -4
data['6M'] = pd.to_numeric(data['6M'])

_9M = data['9M']
for i in range(len(_9M)):
    _9m = str(_9M.iloc[i])
    if _9m[-2:] == 'SD':
        _9M.iloc[i] = -2
    elif _9m[-2:] == 'PR':
        _9M.iloc[i] = -1
    elif _9m[-2:] == 'PD':
        _9M.iloc[i] = -3
    elif _9m[-2:] == 'CR':
        _9M.iloc[i] = 1
    else:
        _9M.iloc[i] = -4
data['9M'] = pd.to_numeric(data['9M'])

Result = data['result']
for i in range(len(_9M)):
    result = str(Result.iloc[i])
    if result == 'SD':
        Result.iloc[i] = -2
    elif result == 'PR':
        Result.iloc[i] = -1
    elif result == 'PD':
        Result.iloc[i] = -3
    elif result == 'CR':
        Result.iloc[i] = 1   
    else:
        Result.iloc[i] = -4
data['result'] = pd.to_numeric(data['result'])

Relapse = data['재발여부']
for i in range(len(Relapse)):
    relapse = Relapse.iloc[i]
    if relapse == 1:
        Relapse[i] = -1
    elif relapse == 0:
        Relapse[i] = 1
    else:
        Relapse[i] = -4
data['재발여부'] = pd.to_numeric(data['재발여부'])

data = data.drop(columns='재발시점')
data = data.drop(columns='사망여부')

sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2, 
                vmax=1, vmin=-1, fmt='1.2f')
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.title('Heat map')
plt.show()
