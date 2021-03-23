import os
import scipy.io
import csv
import numpy as np
import pandas as pd
import pywt

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

def random_batch_sample(data, batch):
    rand_n = np.random.randint(0, len(data))
    while(rand_n > len(data) - batch):
        rand_n = np.random.randint(0, len(data))    
    
    return data[rand_n:rand_n+batch]
    

def generate_sample_data(data_list):
    for data_file in data_list:
        label = data_file[6:-4]
        sample_data = label + '.csv'
        path = './AutoEncoder+Clustering/csv/'
        if os.path.isfile(sample_data):
            sample_dataset = pd.read_csv(sample_data)
        else:
            sample_dataset = pd.DataFrame()
        
        data = pd.read_csv(data_file)
        print(label)
        
        if not os.path.exists(path+label):
            os.makedirs(path+label)
        
        for i in range(101):
            sample_dataset = random_batch_sample(data, 1024)
            
            # 푸리에 변환

            sampling_rate = len(sample_dataset['Drive_End']) * 2
            n = len(sample_dataset['Drive_End'])
        
            F = np.fft.fft(sample_dataset['Drive_End'])/n
            Mag = np.abs(F)
            Pha = np.angle(F)
        
            F2 = np.fft.fft(sample_dataset['Fan_End'])/n
            Mag2 = np.abs(F2)
            Pha2 = np.angle(F2)
            
            length = len(F)
            freq_bin = sampling_rate/length/2
            
            w = np.arange(0, length/2)
            w = w*freq_bin
            
            Mag = Mag[range(int(n/2))]
            Mag2 = Mag2[range(int(n/2))]
            
            plt.title('Frequency Response')
            plt.plot(w, Mag)
            plt.xlabel('Frequency [Hz]')
            plt.show()
            
            plt.title('Frequency2 Response')
            plt.plot(w, Mag2)
            plt.xlabel('Frequency [Hz]')
            plt.show()
            
            if(i < 10):
                number = '_00' + str(i)
            elif(i < 100):
                number = '_0' + str(i)
            else:
                number = '_' + str(i)
                
            file_name = label + number + '.csv'
            
            # coef, freqs = pywt.cwt(sample_dataset['Drive_End'], np.arange(1, 1025), 'morl')
            df = pd.DataFrame(data=Mag, index=None, columns=['DE_fft'])
            df['FA_fft'] = Mag2
            df.to_csv(path + label + '/' + file_name, header=True, index=False)
        
data_list = search('./csv', '.csv')
generate_sample_data(data_list)