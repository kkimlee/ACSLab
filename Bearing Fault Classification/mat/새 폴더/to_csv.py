import os
import scipy.io
import csv
import numpy as np
import pandas as pd

data_list = []
scv_list = []

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



data_list = search('./', '.mat')

'''
DE - drive end accelerometer data
FE - fan end accelerometer data
BA - base accelerometer data
time - time series data
RPM - rpm during testing
'''

for data_file in data_list:
    mat = scipy.io.loadmat(data_file)
    DE = 'X' + list(mat.keys())[7][1:4] + '_DE_time'
    FE = 'X' + list(mat.keys())[7][1:4] + '_FE_time'

    f = open(data_file[2:-4] + '.csv', 'wt', encoding='utf-8', newline="")
    Dwriter = csv.writer(f)
    Dwriter.writerow(['Drive_End', 'Fan_End'])
    
    sensordata = np.concatenate([mat[DE], mat[FE]], axis=1)
    
    for data in sensordata:
        Dwriter.writerow(data)
    
    f.close()

    
