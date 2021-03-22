import os
import csv
import numpy
import pandas as pd

def search(dirname, extension):
    file_list = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        
        if ext==extension:
            file_list.append(full_filename)
            
    return file_list
    

f = open('feature.csv', 'w')

csv_list = search('./', '.csv')


for file in csv_list:
    data = pd.read_csv(file)

