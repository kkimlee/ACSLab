import os
import sys
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

    return file_list

file_list = search('./original/text/', '.txt')
path = './original/csv/'
for file in file_list:
    title = file[2:-4]
    data_row = []
    
    with open(title + '.txt', 'r', encoding='utf-8') as f:
        for row in f:
            data_row.append(row)

    for i in range(len(data_row)):
        if i == 0:
            header = data_row[i].split()
            for j in range(len(header)-1):
                if j == 0:
                    header[j] = header[j] + ' ' + header[j+1]
                else:
                    header[j] = header[j+1]
            del header[len(header)-1]
            sensor_data = pd.DataFrame(columns=header)
        else:
            data = data_row[i].split()
            for k in range(len(data)-1):
                if k == 0:
                    data[k] = data[k] + ' ' + data[k+1]
                else:
                    data[k] = data[k+1]
            del data[len(data)-1]
            sensor_data.loc[i-1] = data
    print(title)
    sensor_data.to_csv(path + title + '.csv', index=False)