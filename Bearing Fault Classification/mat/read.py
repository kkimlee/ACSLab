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

data = []
for data_file in data_list:
    mat = scipy.io.loadmat(data_file)
    print(mat.keys())
    data.append(mat)