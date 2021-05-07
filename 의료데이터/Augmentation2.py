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

from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
from sklearn.decomposition import PCA

import numpy.linalg as lnalg
from scipy.spatial.distance import mahalanobis

import sklearn

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
one_hot_data['age 60'] = label_encoder.fit_transform(np.array(one_hot_age))

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
one_hot_data['MLH1'] = label_encoder.fit_transform(np.array(one_hot_MLH1))

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
one_hot_data['MSH2'] = label_encoder.transform(np.array(one_hot_MSH2))

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
one_hot_data['MSH6'] = label_encoder.transform(np.array(one_hot_MSH6))

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
one_hot_data['PMS2'] = label_encoder.transform(np.array(one_hot_PMS2))

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
one_hot_data['MSI'] = label_encoder.fit_transform(np.array(one_hot_MSI))

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
one_hot_data['3M'] = label_encoder.fit_transform(np.array(one_hot_3m))

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
one_hot_data['6M'] = label_encoder.transform(np.array(one_hot_6m))

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
one_hot_data['9M'] = label_encoder.transform(np.array(one_hot_9m))

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
one_hot_data['result'] = label_encoder.transform(np.array(one_hot_result))

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
data = data.drop(columns='tumor size(mm)')
one_hot_data = one_hot_data.drop(columns='재발시점')
one_hot_data = one_hot_data.drop(columns='사망여부')
one_hot_data = one_hot_data.drop(columns='tumor size(mm)')

one_hot_data = one_hot_data.drop(columns='3M')
one_hot_data = one_hot_data.drop(columns='6M')
one_hot_data = one_hot_data.drop(columns='9M')
one_hot_data = one_hot_data.drop(columns='result')
# one_hot_data = one_hot_data.drop(columns='Relapse')

label = 'Relapse'

cols = one_hot_data.columns.tolist()
cnts = one_hot_data[[label, cols[0]]].groupby(label).count().reset_index()
cnts['delta'] = cnts[cols[0]].max() - cnts[cols[0]]

arr_classes = np.sort(one_hot_data[label].unique())
for cls in arr_classes:
    dlt = int(cnts[cnts[label]==cls]['delta'])
    if dlt <=0: continue
        
    desc_df = one_hot_data[one_hot_data[label]==cls].describe()
    
    sub_arr = np.zeros((dlt, len(cols)-1))
    
    col_counter = 0
    for col in cols[:-1]:
        sub_arr[:, col_counter] = np.random.normal(loc=desc_df[col]['mean'], 
                                                   scale=desc_df[col]['std'], 
                                                   size=dlt
                                                  )
        col_counter += 1
    
    sub_cls = one_hot_data[one_hot_data[label]==cls][label].iloc[0]
    
sub_df = pd.DataFrame(sub_arr, columns=['age 60', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'MSI', 'size'])
sub_df[label] = sub_cls

sub_age = sub_df['age 60']
for i in range(len(sub_age)):
    age = sub_age.iloc[i]
    if age < 0.5:
        sub_age[i] = 0
    else:
        sub_age[i] = 1

sub_MLH1 = sub_df['MLH1']
for i in range(len(sub_MLH1)):
    mlh1 = sub_MLH1.iloc[i]
    if mlh1 < 0.5:
        sub_MLH1[i] = 0
    elif mlh1 < 1.5:
        sub_MLH1[i] = 1
    else:
        sub_MLH1[i] = 2

sub_MSH2 = sub_df['MSH2']
for i in range(len(sub_MSH2)):
    msh2 = sub_MSH2.iloc[i]
    if msh2 < 0.5:
        sub_MSH2[i] = 0
    elif msh2 < 1.5:
        sub_MSH2[i] = 1
    else:
        sub_MSH2[i] = 2
        
sub_MSH6 = sub_df['MSH6']
for i in range(len(sub_MSH6)):
    msh6 = sub_MSH6.iloc[i]
    if msh6 < 0.5:
        sub_MSH6[i] = 0
    elif msh6 < 1.5:
        sub_MSH6[i] = 1
    else:
        sub_MSH6[i] = 2

sub_PMS2 = sub_df['PMS2']
for i in range(len(sub_PMS2)):
    pms2 = sub_PMS2.iloc[i]
    if pms2 < 0.5:
        sub_PMS2[i] = 0
    elif pms2 < 1.5:
        sub_PMS2[i] = 1
    else:
        sub_PMS2[i] = 2

sub_MSI = sub_df['MSI']
for i in range(len(sub_MSI)):
    msi = sub_MSI.iloc[i]
    if msi < 0.5:
        sub_MSI[i] = 0
    elif msi < 1.5:
        sub_MSI[i] = 1
    elif msi < 2.5:
        sub_MSI[i] = 2
    else:
        sub_MSI[i] = 3
        
sub_size = sub_df['size']
for i in range(len(sub_size)):
    size = sub_size.iloc[i]
    if size < 0.5:
        sub_size[i] = 0
    elif size < 1.5:
        sub_size[i] = 1
    elif size < 2.5:
        sub_size[i] = 2
    elif size < 3.5:
        sub_size[i] = 3
    else:
        sub_size[i] = 4
        
data = pd.concat([one_hot_data, sub_df])

label = 'Relapse'
train_data = data.drop(columns = label)
train_label = data[label]

RF_model = RandomForestClassifier()
RF_model.fit(train_data, train_label)
result_RF_model = RF_model.predict(train_data)
RF_model_cm = confusion_matrix(train_label, result_RF_model)
print(RF_model_cm)

GradientBoosting_model = GradientBoostingClassifier()
GradientBoosting_model.fit(train_data, train_label)
result_GradientBoosting_model = GradientBoosting_model.predict(train_data)
GradientBoosting_model_cm = confusion_matrix(train_label, result_GradientBoosting_model)
print(GradientBoosting_model_cm)

loo = LeaveOneOut()
score_RF_loo = cross_val_score(RF_model, train_data, train_label, cv = loo)
score_GB_loo = cross_val_score(GradientBoosting_model, train_data, train_label, cv = loo)

print('score_RF_loo : ', score_RF_loo.mean())
print('score_GB_loo : ', score_GB_loo.mean())

train_data2 = train_data.drop(columns = ['age 60', 'MSH2', 'MSH6', 'MSI', 'size'])

RF_model2 = RandomForestClassifier()
RF_model2.fit(train_data2, train_label)

GB_model2 = GradientBoostingClassifier()
GB_model2.fit(train_data2, train_label)

score_RF_loo2 = cross_val_score(RF_model2, train_data2, train_label, cv = loo)
score_GB_loo2 = cross_val_score(GB_model2, train_data2, train_label, cv = loo)

print('score_RF_loo2 : ', score_RF_loo2.mean())
print('score_GB_loo2 : ', score_GB_loo2.mean())


RF_cross_score = cross_val_score(RF_model2, train_data2, train_label, cv = 12)
GB_cross_score = cross_val_score(GB_model2, train_data2, train_label, cv = 12)

print('RF_cross_score_mean : ', RF_cross_score.mean())
print('GB_cross_score_mean : ', GB_cross_score.mean())

pca_2n = PCA(n_components=2)
pca_3n = PCA(n_components=3)

principal_2n = pca_2n.fit_transform(train_data)
principal_2n_df = pd.DataFrame(principal_2n, columns = ['principal comp 1', 'principal comp 2'])

RF_model3 = RandomForestClassifier()
GB_model3 = GradientBoostingClassifier()

pca_2n_score_RF_loo = cross_val_score(RF_model3, principal_2n_df, train_label, cv = loo)
pca_2n_score_GB_loo = cross_val_score(GB_model3, principal_2n_df, train_label, cv = loo)

print('pca_2n_score_RF_loo : ', pca_2n_score_RF_loo.mean())
print('pca_2n_score_GB_loo : ', pca_2n_score_GB_loo.mean())

pca_2n_RF_cross_score = cross_val_score(RF_model3, principal_2n_df, train_label, cv = 12)
pca_2n_GB_cross_score = cross_val_score(GB_model3, principal_2n_df, train_label, cv = 12)

print('pac_2n_RF_cross_score_mean : ', pca_2n_RF_cross_score.mean())
print('pac_2n_GB_cross_score_mean : ', pca_2n_GB_cross_score.mean())

RF_model4 = RandomForestClassifier()
GB_model4 = GradientBoostingClassifier()

principal_3n = pca_3n.fit_transform(train_data)
principal_3n_df = pd.DataFrame(principal_3n, columns = ['principal comp 1', 'principal comp 2', 'principal comp 3'])

pca_3n_score_RF_loo = cross_val_score(RF_model4, principal_3n_df, train_label, cv = loo)
pca_3n_score_GB_loo = cross_val_score(GB_model4, principal_3n_df, train_label, cv = loo)

print('pca_3n_score_RF_loo : ', pca_3n_score_RF_loo.mean())
print('pca_3n_score_GB_loo : ', pca_3n_score_GB_loo.mean())

pca_3n_RF_cross_score = cross_val_score(RF_model4, principal_3n_df, train_label, cv = 12)
pca_3n_GB_cross_score = cross_val_score(GB_model4, principal_3n_df, train_label, cv = 12)

print('pac_3n_RF_cross_score_mean : ', pca_3n_RF_cross_score.mean())
print('pac_3n_GB_cross_score_mean : ', pca_3n_GB_cross_score.mean())

