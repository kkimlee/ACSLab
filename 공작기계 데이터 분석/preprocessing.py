from openpyxl import load_workbook
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gc
import os

import pandas as pd
from pandas import DataFrame

from scipy.stats import norm

class data_analysis:
    def __init__(self):
        self.feedrate = 960
        self.file_path = './data.xlsx'
    def data_list(self,dl, x, y):
        i = 0
        while dl[x+i][y] != None :
            i += 1
            if x+i == len(dl):
                break
        np_dl = np.array(dl)
        result = np_dl[x:x+i, y:y+3]
        result = result.tolist()
        return result
    
    def get_datalist(self,file_path, feedrate):
        self.feedrate = feedrate
        self.file_path = file_path
        #data_only=Ture로 해줘야 수식이 아닌 값으로 받아온다.
        load_wb = load_workbook(file_path, data_only=True)
        #시트 이름으로 불러오기
        load_ws = load_wb['Sheet1']
         
 
        all_values = []
        for row in load_ws.rows:
            row_value = []
            for cell in row:
                row_value.append(cell.value)
            all_values.append(row_value)
        
        row_num = 0
        for row in all_values :
            col_num = 0
            for cell in row :
                if cell is None :
                    pass
                elif "Machine Condition" in str(cell) :
                    result = [row_num, col_num]
                col_num += 1
            row_num +=1

        
        mc_x = result[0]
        target_row = all_values[mc_x+2]
        counter = 0
        for search in target_row:
            if str(search) == str(self.feedrate) :
                datum = None
                i = 1
                while datum == None :
                    if target_row[counter + i] != None :
                        datum = counter + i
                    i += 1
            counter += 1
        target_data = self.data_list(all_values, mc_x+2, datum)
    
        return target_data
    
    def preprocess(self, df, feedrate):
        
        print(str(feedrate) + '데이터의 결측치 확인')
        print(df.describe())
        
        print(str(feedrate) + '축1spindleLoad 데이터')
        print(df['축1spindleLoad'].describe())
        plt.title(str(feedrate) + 'axis 1 spindleLoad histogram')
        plt.hist(df['축1spindleLoad'])
        plt.xlabel('value')
        plt.ylabel('count')
        plt.show()
        plt.title(str(feedrate) + 'axis 1 spindleLoad boxplot')
        plt.boxplot(df['축1spindleLoad'])
        plt.ylabel('value')
        plt.show()
    
        print(str(feedrate) + '축1 spindleSpeed 데이터')
        print(df['축1 spindleSpeed'].describe())
        plt.title(str(feedrate) + 'axis 1 spindleSpeed histogram')
        plt.hist(df['축1 spindleSpeed'])
        plt.xlabel('value')
        plt.ylabel('count')
        plt.show()
        plt.title(str(feedrate) + 'axis 1 spindleSpeed boxplot')
        plt.boxplot(df['축1 spindleSpeed'])
        plt.ylabel('value')
        plt.show()
    
        print(str(feedrate) + 'Feed')
        print(df['Feed'].describe())
        plt.title(str(feedrate) + 'Feed histogram')
        plt.hist(df['Feed'])
        plt.xlabel('value')
        plt.ylabel('count')
        plt.show()
        plt.title(str(feedrate) + 'feed boxplot')
        plt.boxplot(df['Feed'])
        plt.ylabel('value')
        plt.show()
        
    
        spindleSpeed_outliers = df[4790 > df['축1 spindleSpeed']].index
        df = df.drop(spindleSpeed_outliers)
        spindleSpeed_outliers = df[4810 < df['축1 spindleSpeed']].index
        df = df.drop(spindleSpeed_outliers)
        # spindleSpeed_outliers = df[4800 != df['축1 spindleSpeed']].index
        # df = df.drop(spindleSpeed_outliers)
    
        Feed_outliers = df[(feedrate-10) > df['Feed']].index
        df = df.drop(Feed_outliers)
        Feed_outliers = df[(feedrate+10) < df['Feed']].index
        df = df.drop(Feed_outliers)
        # Feed_outliers = df[Feed != df['Feed']].index
        # df = df.drop(Feed_outliers)
    
        
        print(str(feedrate) + '축1spindleLoad 데이터')
        print(df['축1spindleLoad'].describe())
        plt.title(str(feedrate) + 'axis 1 spindleLoad histogram')
        plt.hist(df['축1spindleLoad'])
        plt.xlabel('value')
        plt.ylabel('count')
        plt.show()
        plt.title(str(feedrate) + 'axis 1 spindleLoad boxplot')
        plt.boxplot(df['축1spindleLoad'])
        plt.ylabel('value')
        plt.show()
        
        print(str(feedrate) + '축1 spindleSpeed 데이터')
        print(df['축1 spindleSpeed'].describe())
        plt.title(str(feedrate) + 'axis 1 spindleSpeed histogram')
        plt.hist(df['축1 spindleSpeed'])
        plt.xlabel('value')
        plt.ylabel('count')
        plt.show()
        plt.title(str(feedrate) + 'axis 1 spindleSpeed boxplot')
        plt.boxplot(df['축1 spindleSpeed'])
        plt.ylabel('value')
        plt.show()
        
        print(str(feedrate) + 'Feed')
        print(df['Feed'].describe())
        plt.title(str(feedrate) + 'Feed histogram')
        plt.hist(df['Feed'])
        plt.xlabel('value')
        plt.ylabel('count')
        plt.show()
        plt.title(str(feedrate) + 'feed boxplot')
        plt.boxplot(df['Feed'])
        plt.ylabel('value')
        plt.show()
    
        print(df.corr())
        sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2, 
                vmax=1, vmin=-1, fmt='1.2f')
        fig=plt.gcf()
        fig.set_size_inches(8,6)
        plt.title(str(feedrate) + 'Heat map')
        plt.show()
        
        plt.plot(df['축1spindleLoad'])
        plt.title(str(feedrate) + 'axis 1 spindleLoad')
        plt.ylabel('value')
        plt.show()
        
        plt.plot(df['축1 spindleSpeed'])
        plt.title(str(feedrate) + 'axis 1 spindleSpeed')
        plt.ylabel('value')
        plt.show()
        
        plt.plot(df['Feed'])
        plt.title(str(feedrate) + 'Feed')
        plt.ylabel('value')
        plt.show()
        
        return df
    
    def magic_feature(self, df, feature):
        
        # feature의 최대 값
        max_value = df[feature].max()
        
        # feature의 최소 값
        min_value = df[feature].min()
        
        # feature의 최대 값 - 최소 값
        peak_value = max_value - min_value
        
        # feature의 표준 편차
        std_value = df[feature].std()
        
        # feature의 skew
        skew_value = df[feature].skew()
        
        # feature의 kurt
        kurt_value = df[feature].kurt()
        
        # feature의 절대 값
        abs_value = df[feature].abs()
        
        # feature의 절대 값의 평균 값
        absMean_value = abs_value.mean()
        
        # feature의 제곱의 루트 값
        root_square_value = (df[feature])**0.5
        
        # feature의 제곱의 루트 값의 평균 값
        sqr_amp_value = (root_square_value.mean())**2
        
        # feature의 제곱의 루트 값을 feature의 절대 값 평균으로 나눈 값
        shape_factor_value = (((df[feature]**2).mean())**0.5) / abs_value.mean()
        
        magic_feature = [max_value, min_value, peak_value, std_value, skew_value, kurt_value, absMean_value, sqr_amp_value, shape_factor_value]
        magic_feature = DataFrame(magic_feature, ['max_value', 'min_value', 'peak_value', 'std_value', 'skew_value', 'kurt_value', 'absMean_value', 'sqr_amp_value', 'shape_factor_value'])
        
        print('중앙 값 :',  df[feature].median)
        
        return magic_feature
    
    
    def probability_distribution(self, df, feature, feedrate):
        
        x = df[feature]
        x = x.sort_values()
        y = (1 / (np.sqrt(2 * np.pi) * x.var())) * np.exp(-1 * ((x-x.std()) ** 2 / (2 * (x.var()**2))))
        plt.title(str(feedrate) + ' Feed ' + feature + ' probability distribution')
        plt.plot(x, y)
        plt.xlabel('value')
        plt.show()
        

        return x, y
    
if __name__ == '__main__':
    # 데이터 읽어오기
    data_anal = data_analysis()
    data_960 = data_anal.get_datalist(file_path = './data.xlsx', feedrate=960)
    data_2000 = data_anal.get_datalist(file_path = './data.xlsx', feedrate=2000)
    data_2200 = data_anal.get_datalist(file_path = './data.xlsx', feedrate=2200)
    data_2400 = data_anal.get_datalist(file_path = './data.xlsx', feedrate=2400)
    data_2600 = data_anal.get_datalist(file_path = './data.xlsx', feedrate=2600)
    
    
    # 데이터 전처리
    df_960 = DataFrame(data_960, columns = ['축1spindleLoad', '축1 spindleSpeed', 'Feed'])
    df_960 = data_anal.preprocess(df_960, 960)
    df_960_magic_feature = data_anal.magic_feature(df_960, '축1spindleLoad')
    print("df_960")
    print(df_960_magic_feature)
    
    df_2000 = DataFrame(data_2000, columns = ['축1spindleLoad', '축1 spindleSpeed', 'Feed'])
    df_2000 = data_anal.preprocess(df_2000, 2000)
    df_2000_magic_feature = data_anal.magic_feature(df_2000, '축1spindleLoad')
    print("df_2000")
    print(df_2000_magic_feature)
    
    df_2200 = DataFrame(data_2200, columns = ['축1spindleLoad', '축1 spindleSpeed', 'Feed'])
    df_2200 = data_anal.preprocess(df_2200, 2200)
    df_2200_magic_feature = data_anal.magic_feature(df_2200, '축1spindleLoad')
    print("df_2200")
    print(df_2200_magic_feature)
    
    df_2400 = DataFrame(data_2400, columns = ['축1spindleLoad', '축1 spindleSpeed', 'Feed'])
    df_2400 = data_anal.preprocess(df_2400, 2400)
    df_2400_magic_feature = data_anal.magic_feature(df_2400, '축1spindleLoad')
    print("df_2400")
    print(df_2400_magic_feature)
    
    df_2600 = DataFrame(data_2600, columns = ['축1spindleLoad', '축1 spindleSpeed', 'Feed'])
    df_2600 = data_anal.preprocess(df_2600, 2600)
    df_2600_magic_feature = data_anal.magic_feature(df_2600, '축1spindleLoad')
    print("df_2600")
    print(df_2600_magic_feature)
    
    x, y = data_anal.probability_distribution(df_2000, '축1spindleLoad', 2000)