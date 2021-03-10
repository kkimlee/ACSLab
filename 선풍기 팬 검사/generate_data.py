import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Input, Flatten, concatenate


def search(dirname, extension):
    file_list = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == extension:
            file_list.append(full_filename)
            print(full_filename)

    return file_list

def random_batch_sample(data, batch):
    rand_n = np.random.randint(0, len(data))
    while(rand_n > len(data) - batch):
        rand_n = np.random.randint(0, len(data))    

    return data[rand_n:rand_n+batch]

file_list = search('./', '.csv')


for file in file_list:
    df = pd.read_csv(file)
    
    for i in range(120):
        tmp_df = random_batch_sample(df, 1000)
        
        if i < 10:
            label = '00' + str(i)
        elif i < 100:
            label = '0' + str(i)
        else:
            label = str(i)
        
        if 'motor' in file:
            if 'abnormal' in file:
                tmp_df.to_csv('data/motor_abnormal/motor_abnormal ' + label + '.csv', index=False)
            elif 'normal' in  file:
                tmp_df.to_csv('data/motor_normal/motor_normal ' + label + '.csv', index=False)
        elif 'cover' in file:
            if 'abnormal' in file:
                tmp_df.to_csv('data/cover_abnormal/cover_abnormal ' + label + '.csv', index=False)
            elif 'normal' in file:
                tmp_df.to_csv('data/cover_normal/cover_normal ' + label + '.csv', index=False)
        
    
    

