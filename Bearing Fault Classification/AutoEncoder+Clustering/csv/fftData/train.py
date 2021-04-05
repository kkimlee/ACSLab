import os
import csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models, utils
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Input, Flatten, concatenate, LeakyReLU, Dropout, Reshape


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Standardization 평균 0 / 분산 1
scaler = StandardScaler()

# Normalization 최소값 0 / 최대값 1
# scaler = MinMaxScaler()

def search(dirname, extension):
    file_list = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == extension and extension != '':
            file_list.append(full_filename)
        else:
            file_list.append(full_filename)
            
    return file_list

C1_category = search('./train/weight_0', '')
len_category = len(C1_category)

C1_all = list()
C1_sp = list()
C1_un = list()
C1_un_label = list()

C1_all_file = list()
C1_sp_file = list()
C1_un_file = list()

for idx in range(len_category):
    for i in range(4):
        C1_category = search('./train/weight_' + str(i), '')[idx]
        
        C1_data = search(C1_category, '.csv')
        for file in C1_data:
            data = pd.read_csv(file)['fft']
            C1_all.append(data)
            C1_all_file.append(file)
            
        C1_sp.append(pd.read_csv(C1_data[0])['fft'])
        C1_sp_file.append(C1_data[0])
        for file in C1_data[1:]:
            C1_un.append(pd.read_csv(file)['fft'])
            C1_un_file.append(file)
            C1_un_label.append(idx)

centroid_label = list()
for i in range(10):
    centroid_label.append(i)

''' 
i=0
for data_dir in C1_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['fft']
        C1_all.append(data)
        
    C1_sp.append(pd.read_csv(C1_data[0])['fft'])
    
    for file in C1_data[1:]:
        C1_un.append(pd.read_csv(file)['fft'])
        C1_un_label.append(i)
    i += 1
    
C1_category = search('./train/weight_1', '')
i=0
for data_dir in C1_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['fft']
        C1_all.append(data)
        
    C1_sp.append(pd.read_csv(C1_data[0])['fft'])
    for file in C1_data[1:]:
        C1_un.append(pd.read_csv(file)['fft'])
        C1_un_label.append(i)
    i += 1

C1_category = search('./train/weight_2', '')
i=0
for data_dir in C1_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['fft']
        C1_all.append(data)
        
    C1_sp.append(pd.read_csv(C1_data[0])['fft'])
    for file in C1_data[1:]:
        C1_un.append(pd.read_csv(file)['fft'])
        C1_un_label.append(i)
    i += 1

C1_category = search('./train/weight_3', '')
i=0
for data_dir in C1_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['fft']
        C1_all.append(data)
        
    C1_sp.append(pd.read_csv(C1_data[0])['fft'])
    for file in C1_data[1:]:
        C1_un.append(pd.read_csv(file)['fft'])
        C1_un_label.append(i)
    i += 1
'''
C1_all = np.array(C1_all)[:, :]
C1_sp = np.array(C1_sp)[:, :]
C1_un = np.array(C1_un)[:, :]

encoder_input = Input(shape=(512,))
encoded = Dense(320, activation=LeakyReLU())(encoder_input)
encoded = Dense(160, activation=LeakyReLU())(encoder_input)
encoded = Dense(80, activation=LeakyReLU())(encoder_input)
encoded = Dense(128)(encoder_input)

decoded = Dense(80,  activation=LeakyReLU())(encoded)
decoded = Dense(160,  activation=LeakyReLU())(encoded)
decoded = Dense(320,  activation=LeakyReLU())(encoded)
decoded = Dense(512,  activation=LeakyReLU())(encoded)

autoencoder = Model(encoder_input, decoded)

encoder = Model(encoder_input, encoded)

decoder_input = Input(shape=(128,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(decoder_input, decoder_layer(decoder_input))

autoencoder.compile(optimizer=Adam(lr=0.0001),
              loss='mae')

autoencoder.summary()

history = autoencoder.fit(C1_all, C1_all,
       batch_size=64,
       epochs=1000,
       validation_data=(C1_all, C1_all))

def encoder_loss(ae_loss, inter_loss):
    def loss(y_true, y_pred):
        # y_pred = scaler.fit_transform(y_pred)
        cluster_loss = K.mean(K.abs((y_true - y_pred)) + abs(ae_loss) - abs(inter_loss))

        return cluster_loss
    return loss
    
tsne = TSNE(n_components=2, random_state=0)
lle = LocallyLinearEmbedding(n_components=2, random_state=0)
kpca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04, random_state=0)
isomap = Isomap(n_components=2)

for epi in range(15):   
    sp_latent_result = encoder.predict(C1_sp)
    # sp_latent_result = scaler.fit_transform(sp_latent_result)
    sp_latent_mean = 0
    sp_latent = list()
    for i in range(len(sp_latent_result)):
        if i%4 == 3:
            sp_latent_mean += sp_latent_result[i]
            sp_latent_mean = sp_latent_mean/4
            sp_latent.append(sp_latent_mean)
            sp_latent_mean = 0
        else:
            sp_latent_mean += sp_latent_result[i]
            
    sp_centroid = list()
    abs_sp_centroid = list()
    for i in range(len(sp_latent)):
        abs_sp_centroid.append(sp_latent[i])
        for j in range(400):
            sp_centroid.append(sp_latent[i])
    
    inter_loss = 0
    for i in range(len(abs_sp_centroid)-1):
        inter_loss += np.mean(abs_sp_centroid[i]-abs_sp_centroid[i+1])
    inter_loss = inter_loss/(len(abs_sp_centroid)-1)
    
    sp_centroid = np.array(sp_centroid)
    sp_centorid = np.array(sp_centroid)
    
    for epoch in range(70):
        ae_loss = autoencoder.fit(C1_un, C1_un, batch_size=32, epochs=1).history['loss'][0]
        # ae_loss = 0
        encoder.compile(optimizer=Adam(lr=0.0001),
                       loss=encoder_loss(ae_loss=ae_loss, inter_loss=inter_loss))
        
        # encoder.compile(optimizer=Adam(lr=0.0001),
        #                 loss='mae')
        
        encoder.fit(C1_un, sp_centroid,
                    batch_size=128,
                    epochs=1)
    
    
    un_latent = encoder.predict(C1_un)
    # un_latent = np.array(un_latent)[:, :]
    
    np_abs_sp_centroid = np.array(abs_sp_centroid)
    test = np.concatenate((np.array(abs_sp_centroid), un_latent), axis=0)
    
    test_results = tsne.fit_transform(test)
    check = test_results[:10]
    test_centroid = pd.DataFrame(test_results[:10], columns=['first', 'second'])
    test_un = pd.DataFrame(test_results[10:], columns=['first', 'second'])
    
    df_test_centroid_label = pd.DataFrame(centroid_label, columns=['label'])
    df_test_un_label = pd.DataFrame(C1_un_label, columns=['label'])
    
    test_un['label'] = df_test_un_label
    test_centroid['label'] = df_test_centroid_label
    
    plt.scatter(test_un['first'], test_un['second'], c=test_un['label'], alpha=0.4, label='un')
    plt.scatter(test_centroid['first'], test_centroid['second'], c=test_centroid['label'], marker='s', s=250, label='sp')
    
    plt.title('TSNE')
    plt.show()
    
    '''
    T-sne
    '''
    # tsne_results = un_tsne.fit_transform(un_latent)
    # tsne_centroid = sp_tsne.fit_transform(abs_sp_centroid)

    # tsne_results = tsne.fit_transform(un_latent)
    # tsne_centroid = tsne.fit_transform(abs_sp_centroid)
    
    # tsne_results = pd.DataFrame(tsne_results, columns=['first', 'second'])
    # tsne_centroid = pd.DataFrame(tsne_centroid, columns=['first', 'second'])
    
    # tsne_results = pd.DataFrame(tsne_results, columns=['first', 'second', 'third'])
    # tsne_centroid = pd.DataFrame(tsne_centroid, columns=['first', 'second', 'third'])
    
    # df_tsne_un_label = pd.DataFrame(C1_un_label, columns=['label'])
    # df_tsne_centroid_label = pd.DataFrame(centroid_label, columns=['label'])
    
    # tsne_results['label'] = df_tsne_un_label
    # tsne_centroid['label'] = df_tsne_centroid_label
    
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    # plt.scatter(tsne_results['first'], tsne_results['second'], c=tsne_results['label'], alpha=0.4, label='un')
    # plt.scatter(tsne_centroid['first'], tsne_centroid['second'], c=tsne_centroid['label'], marker='s', s=250, label='sp')
    
    # ax.scatter(tsne_results['first'], tsne_results['second'], tsne_results['third'], c=tsne_results['label'])
    # ax.scatter(tsne_centroid['first'], tsne_centroid['second'], tsne_centroid['third'], c=tsne_centroid['label'])
    # plt.title('T-SNE results')
    # plt.legend()
    # plt.show()
    
    '''
    LLE
    '''
    # lle_results = lle.fit_transform(un_latent)
    # lle_centroid = lle.fit_transform(abs_sp_centroid)
    
    # lle_results = pd.DataFrame(lle_results, columns=['first', 'second'])
    # lle_centroid = pd.DataFrame(lle_centroid, columns=['first', 'second'])
    
    # df_lle_un_label = pd.DataFrame(C1_un_label, columns=['label'])
    # df_lle_centroid_label = pd.DataFrame(centroid_label, columns=['label'])
    
    # lle_results['label'] = df_lle_un_label
    # lle_centroid['label'] = df_lle_centroid_label
    
    # plt.scatter(lle_results['first'], lle_results['second'], c=lle_results['label'], alpha=0.4, label='un')
    # plt.scatter(lle_centroid['first'], lle_centroid['second'], c=lle_centroid['label'], marker='s', s=250, label='sp')
    
    # plt.title('LLE results')
    # plt.legend()
    # plt.show()
    
    
    '''
    KPCA
    '''
    # kpca_results = kpca.fit_transform(un_latent)
    # kpca_centroid = kpca.fit_transform(abs_sp_centroid)
    
    # kpca_results = pd.DataFrame(kpca_results, columns=['first', 'second'])
    # kpca_centroid = pd.DataFrame(kpca_centroid, columns=['first', 'second'])
    
    # df_kpca_un_label = pd.DataFrame(C1_un_label, columns=['label'])
    # df_kpca_centroid_label = pd.DataFrame(centroid_label, columns=['label'])
    
    # kpca_results['label'] = df_kpca_un_label
    # kpca_centroid['label'] = df_kpca_centroid_label
    
    # plt.scatter(kpca_results['first'], kpca_results['second'], c=kpca_results['label'], alpha=0.4, label='un')
    # plt.scatter(kpca_centroid['first'], kpca_centroid['second'], c=kpca_centroid['label'], marker='s', s=250, label='sp')
    
    # plt.title('KPCA results')
    # plt.legend()
    # plt.show()
    
    '''
    ISO MAP
    '''
    # isomap_results = isomap.fit_transform(un_latent)
    # isomap_centroid = isomap.fit_transform(abs_sp_centroid)
    
    # isomap_results = pd.DataFrame(isomap_results, columns=['first', 'second'])
    # isomap_centroid = pd.DataFrame(isomap_centroid, columns=['first', 'second'])
    
    # df_isomap_un_label = pd.DataFrame(C1_un_label, columns=['label'])
    # df_isomap_centroid_label = pd.DataFrame(centroid_label, columns=['label'])
    
    # isomap_results['label'] = df_isomap_un_label
    # isomap_centroid['label'] = df_isomap_centroid_label
    
    # plt.scatter(isomap_results['first'], isomap_results['second'], c=isomap_results['label'], alpha=0.4, label='un')
    # plt.scatter(isomap_centroid['first'], isomap_centroid['second'], c=isomap_centroid['label'], marker='s', s=250, label='sp')
    
    # plt.title('ISOMAP results')
    # plt.legend()
    # plt.show()
    
# classifiered_input = Input(shape=(128,))
classifiered = Dense(320, activation=LeakyReLU())(encoded)
classifiered = Dense(160, activation=LeakyReLU())(classifiered)
classifiered = Dense(128, activation=LeakyReLU())(classifiered)
classifiered = Dense(10, activation='softmax')(classifiered)

classifier = Model(encoder_input, classifiered)

classifier.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

label_encoder = LabelEncoder()

C1_un_label = label_encoder.fit_transform(np.array(C1_un_label))
C1_un_label = utils.to_categorical(C1_un_label, 10)

C1_test_category = search('./test/weight_0', '')

classifier.fit(C1_un, C1_un_label, batch_size=32, epochs=400)

C1_test = list()
C1_test_label = list()
i = 0
for data_dir in C1_test_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['DE_fft']
        C1_test.append(data)
        C1_test_label.append(i)
    i += 1

C1_test_category = search('./test/weight_1', '')
i = 0
for data_dir in C1_test_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['DE_fft']
        C1_test.append(data)
        C1_test_label.append(i)
    i += 1

C1_test_category = search('./test/weight_2', '')
i = 0
for data_dir in C1_test_category:
    C1_data = search(data_dir, '.csv')
    for file in C1_data:
        data = pd.read_csv(file)['DE_fft']
        C1_test.append(data)
        C1_test_label.append(i)
    i += 1

C1_test = np.array(C1_test)[:, :]
    
C1_test_label = label_encoder.fit_transform(np.array(C1_test_label))
C1_test_label = utils.to_categorical(C1_test_label, 10)

test_loss, test_acc = classifier.evaluate(C1_test, C1_test_label)
print(test_acc)

encoder.save('encoder.h5')
decoder.save('decoder.h5')
classifier.save('classifier.h5')