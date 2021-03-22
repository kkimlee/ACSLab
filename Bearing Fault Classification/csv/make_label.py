import os
import scipy.io
import csv
import numpy as np
import pandas as pd


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

data_list = search('./', '.csv')

# 라벨 만들기
for file in data_list:
    data = pd.read_csv(file[2:])
    data['Label'] = file[2:-6]
    data.to_csv(file[2:], header=True, index=False)


    

    


