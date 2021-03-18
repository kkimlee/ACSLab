import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Input, Flatten, concatenate


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

file_list = search('./original/csv/', '.csv')


for file in file_list:
    df = pd.read_csv(file)
    
    name = file.split('/')[-1]
    name = name.split('.')[0]
    
    train_path = 'train/'
    if not os.path.exists(train_path + name):
        os.makedirs(train_path + name)
    
    test_path = 'test/'
    if not os.path.exists(test_path + name):
        os.makedirs(test_path + name)
        
    for i in range(120):
        train_df = random_batch_sample(df, 10)
        test_df = random_batch_sample(df, 10)
        
        if i < 10:
            label = '00' + str(i)
        elif i < 100:
            label = '0' + str(i)
        else:
            label = str(i)
        
        train_df.to_csv(train_path + name + '/' + label + '.csv', index=False)
        test_df.to_csv(test_path + name + '/' + label + '.csv', index=False)
        
        print(name + '/' + label)
    
    

