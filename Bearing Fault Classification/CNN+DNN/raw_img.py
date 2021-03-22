import os
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

def generate_img(data_list):
    
    for data in data_list:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
        if os.path.isfile(data):
            sample_dataset = pd.read_csv(data, float_precision='high')
        else:
            sample_dataset = pd.DataFrame()
    
        x = sample_dataset['Drive_End']
        print(data[2:-4])
        
        ax.plot(x)
        ax.axis('off')
        plt.savefig(data[2:-4] + '.png')
        plt.show()
    

data_list = search('./', '.csv')
generate_img(data_list)