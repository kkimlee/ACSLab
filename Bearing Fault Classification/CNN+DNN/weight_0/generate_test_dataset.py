import os
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

BF_1 = search('./csv/BF/0.18mm/', '.csv')
BF_2 = search('./csv/BF/0.36mm/', '.csv')
BF_3 = search('./csv/BF/0.54mm/', '.csv')
IF_1 = search('./csv/IF/0.18mm/', '.csv')
IF_2 = search('./csv/IF/0.36mm/', '.csv')
IF_3 = search('./csv/IF/0.54mm/', '.csv')
OF_1 = search('./csv/OF/0.18mm/', '.csv')
OF_2 = search('./csv/OF/0.36mm/', '.csv')
OF_3 = search('./csv/OF/0.54mm/', '.csv')
NO = search('./csv/NO/', '.csv')

State = [BF_1, BF_2, BF_3, IF_1, IF_2, IF_3, OF_1, OF_2, OF_3, NO]

f = open('test.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(['max', 'min', 'peak', 'std', 'skew', 'kurt', 'absMean', 'sqr_amp', 'shape_factor', 'img_path', 'raw_path', 'label'])


for St in State:
    for file in St[200:]:
        data = pd.read_csv(file, float_precision='high')
        max_value = data['Drive_End'].max()
        min_value = data['Drive_End'].min()
        peak_value = max_value - min_value
        std_value = data['Drive_End'].std()
        skew_value = data['Drive_End'].skew()
        kurt_value = data['Drive_End'].kurt()
        abs_value = data['Drive_End'].abs()
        absMean_value = abs_value.mean()
        root_square_value = (data['Drive_End']**2)**0.5
        sqr_amp_value = root_square_value.mean()
        shape_factor_value = (((data['Drive_End']**2).mean())**0.5) / abs_value.mean()
        img_path = './img/' + file[6:-4] + '.png'
        raw_path = './csv/' + file[6:-4] + '.csv'
        label = data['Label'][0]
        wr.writerow([max_value, min_value, peak_value, std_value, skew_value, kurt_value, absMean_value, sqr_amp_value, shape_factor_value, img_path, raw_path, label])
        
f.close()