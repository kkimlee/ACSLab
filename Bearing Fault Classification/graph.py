import os
import scipy.io
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
        
    return file_list

        
data_list = search('./csv', '.csv')

for file in data_list:
    data = pd.read_csv(file)
    
    DE = data['Drive_End']
    FE = data['Fan_End']
    
    plt.plot(DE, label = 'Drive_End')
    plt.plot(FE, label = 'Fan_End')
    
    plt.xlabel('Numbers')
    plt.ylabel('Value')
    
    plt.ylim(-7.0, 7.0)
    
    plt.title(file[6:-4])
    
    plt.legend()
    
    plt.savefig('fig/'+file[6:-4]+'.png')
    plt.show()
    