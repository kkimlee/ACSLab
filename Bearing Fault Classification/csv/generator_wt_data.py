import os
import pywt
import scipy
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


def search(dirname, extension):
    file_list = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == extension:
            file_list.append(full_filename)
            print (full_filename)
        
    return file_list

def random_batch_sample(data, batch):
    rand_n = np.random.randint(0, len(data))
    while(rand_n > len(data) - batch):
        rand_n = np.random.randint(0, len(data))    
    
    return data[rand_n:rand_n+batch]

def generate_sample_data(data_list):
    for data_file in data_list:
        label = data_file[2:-6]
        sample_data = label + '.csv'
        
        pd.set_option('precision', 20)
    
        if os.path.isfile(sample_data):
            sample_dataset = pd.read_csv(sample_data, float_precision='high')
        else:
            sample_dataset = pd.DataFrame()
        
        data = pd.read_csv(data_file, float_precision='high')
    
        for i in range(50, 100):
            sample_dataset = random_batch_sample(data, 1024)

            coef, freqs = pywt.cwt(sample_dataset['Drive_End'], np.arange(1, 1025), 'morl')
            
            coef = np.where(coef > 0, coef, 0)
            coef = scipy.misc.imresize(coef, (32, 32), interp='bicubic')
            
            '''
            coef = cv2.resize(coef, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
            coef = cv2.resize(coef, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            coef = cv2.resize(coef, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            coef = cv2.resize(coef, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            coef = cv2.resize(coef, dsize=(32,32), interpolation=cv2.INTER_CUBIC)
            '''
            
            plt.matshow(coef, cmap='gray')
            plt.imsave('../wt_train/' + data_file[2:-4] +'_' + str(i) + '.png', coef, cmap='gray')
            plt.show()
            
data_list = search('./', '.csv')
generate_sample_data(data_list)