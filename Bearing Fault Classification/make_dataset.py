import os
import scipy.io
import csv
import numpy as np
import pandas as pd
import pywt

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
    
        if os.path.isfile(sample_data):
            sample_dataset = pd.read_csv(sample_data)
        else:
            sample_dataset = pd.DataFrame()
        
        data = pd.read_csv(data_file)
        print(label)
        
        for i in range(237):
            sample_dataset = random_batch_sample(data, 512)
            
            if(i < 10):
                number = '_00' + str(i)
            elif(i < 100):
                number = '_0' + str(i)
            else:
                number = '_' + str(i)
                
            file_name = label + number + '.csv'
            # coef, freqs = pywt.cwt(sample_dataset['Drive_End'], np.arange(1, 1025), 'morl')
            sample_dataset.to_csv('./raw_data/' + file_name, header=True, index=False)
        
data_list = search('./csv', '.csv')
generate_sample_data(data_list)