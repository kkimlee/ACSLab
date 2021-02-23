import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

file_list = search('./', '.csv')

data = []
for file in file_list:
    data.append(pd.read_csv(file))


for i in range(len(data)):
    title_name = ['acc', 'gyro', 'angle']
    data_name = ['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'AngleX', 'AngleY', 'AngleZ']
    for j in range(3):
        fig = plt.figure()
        plt.title(file_list[i][2:-4] + ' ' + title_name[j])
        
        ax = fig.add_subplot(1, 1, 1)
        
        ax.plot(data[i][data_name[j*3]])
        ax.plot(data[i][data_name[j*3 + 1]])
        ax.plot(data[i][data_name[j*3 + 2]])
        
        plt.legend(labels=[data_name[j*3], data_name[j*3 + 1], data_name[j*3 + 2]])
        plt.show()
    
    
    