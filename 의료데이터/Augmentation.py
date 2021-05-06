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

import numpy.linalg as lnalg
from scipy.spatial.distance import mahalanobis

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

# one_hot_data = one_hot_data.drop(columns='3M')
one_hot_data = one_hot_data.drop(columns='6M')
one_hot_data = one_hot_data.drop(columns='9M')
one_hot_data = one_hot_data.drop(columns='result')
one_hot_data = one_hot_data.drop(columns='Relapse')

label = '3M'

one_hot_data = one_hot_data.drop(one_hot_data.index[1])
arr_classes = np.sort(one_hot_data[label].unique())
gmms = dict()
for cls in arr_classes:
    print('class', cls)
    
    # extract samples for the class and convert to numpy
    # and remove the class label which is the last column in the dataframe
    x_train = one_hot_data[one_hot_data[label] == cls].values[:, :-1]
    
    # train gmm, extract results, and save in dictionary
    gmm = GMM(n_components=1, covariance_type = 'full').fit(x_train)
    gmms[cls-1] = (gmm.means_, gmm.covariances_)
    
for cls in arr_classes:
    # let's randomly select 1000 samples
    sub_sample = one_hot_data[one_hot_data[label]==cls].sample(frac=1).values[:1000, :-1]
    
    mh = np.empty((sub_sample.shape[0]))
    mu, sig = gmms[cls-1]
    # we have to invert the matrix for mahalanobis calc
    isig = lnalg.inv(sig)
    
    for i in range(sub_sample.shape[0]):
        mh[i] = mahalanobis(sub_sample[i], mu, isig)
        
    print('class', cls, 'mean', mh.mean())
    
inv_sig = dict()
mh = np.zeros((arr_classes.shape[0]))

smpls = one_hot_data.sample(frac=1).values[:1000]
x = smpls[:, :-1]
labels = smpls[:, -1:]
results = np.zeros(x.shape[0])

# let's invert all sigmas from GMMs
for cls in arr_classes:
    mu,sig = gmms[cls-1]
    isig = lnalg.inv(sig)
    inv_sig[cls-1] = mu, isig

for i in range(x.shape[0]):
    for cls in arr_classes:
        mu, isig = inv_sig[cls-1]
        mh[cls-1] = mahalanobis(x[i], mu, isig)
       
    # if gmm prediction matches the original class label let's save the result
    if np.argmin(mh) == labels[i]-1:
        results[i] = 1
        
# let's calculate simple accuracy: True / All Samples
acc = results.sum() / results.shape[0]

print('Accuracy:', acc)
if acc > .9: print('Doing well')

cols = one_hot_data.columns.tolist()
trg = .4
cnts = one_hot_data[[label, cols[0]]].groupby(label).count().reset_index()
print(cnts[cols[0]].max())
cnts['delta'] = cnts[cols[0]].max() - cnts[cols[0]]

# NOTE: let's create a list of categorical columns so we can normalize
# them to 0 and 1 values after synthesizing new data

# let's see how many unique values each column has
cat_columns = [len(one_hot_data[col].unique()) for col in one_hot_data.columns]

# now let's assume that columns with only two unique values
# are categorical - typically true for one-hot encoded datasets
cat_columns =[0 if x > 2 else 1 for x in cat_columns]

# generate a new distribution for each column and class
new_class_arrays = []

for cls in arr_classes:
    dlt = int(cnts[cnts[label]==cls]['delta'])
    if dlt <=0: continue
        
    desc_df = one_hot_data[one_hot_data[label]==cls].describe()
    
    sub_arr = np.zeros((dlt, len(cols), 1))
    
    col_counter = 0
    for col in cols:
        sub_arr[:, col_counter] = np.random.normal(loc=desc_df[col]['mean'], 
                                                   scale=desc_df[col]['std'], 
                                                   size= (dlt, 1)
                                                  )
        col_counter += 1
    new_class_arrays.append(sub_arr)

new_samples = np.concatenate(new_class_arrays)
# now that we have our new samples let's convert the purturbed columns back
# to categorical columns using the list that we have created above

c_idx = 0
for c in cat_columns:
    if c == 1:
        # get slices, then update them which updates the parent array
        s = new_samples[:, c_idx, :]
        s[s < .5] = 0
        s[s >=.5] = 1
    c_idx +=1


test = new_samples
print(test)
x = new_samples[:, :-1]
labels = new_samples[:, -1:]
results = np.zeros(x.shape[0])


for i in range (x.shape[0]):
    for cls in arr_classes:
        mu, isig = inv_sig[cls-1]
        mh[cls-1] = mahalanobis(x[i], mu, isig)
       
    # if gmm assignment is not the same as the original label then either discard or re-assign
    # we are going to re-assign the class
    if np.argmin(mh) != labels[i]-1:
        labels[i] = np.argmin(mh)