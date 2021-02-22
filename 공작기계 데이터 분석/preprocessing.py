from openpyxl import load_workbook
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gc
import os

import pandas as pd
from pandas import DataFrame

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
        '''
        print('축1spindleLoad 데이터')
        print(df['축1spindleLoad'].describe())
        plt.title('axis 1 spindleLoad histogram')
        plt.hist(df['축1spindleLoad'])
        plt.show()
        plt.title('axis 1 spindleLoad boxplot')
        plt.boxplot(df['축1spindleLoad'])
        plt.show()
    
        print('축1 spindleSpeed 데이터')
        print(df['축1 spindleSpeed'].describe())
        plt.title('axis 1 spindleSpeed histogram')
        plt.hist(df['축1 spindleSpeed'])
        plt.show()
        plt.title('axis 1 spindleSpeed boxplot')
        plt.boxplot(df['축1 spindleSpeed'])
        plt.show()
    
        print('Feed')
        print(df['Feed'].describe())
        plt.title('Feed histogram')
        plt.hist(df['Feed'])
        plt.show()
        plt.title('feed boxplot')
        plt.boxplot(df['Feed'])
        plt.show()
        '''
    
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
    
        print('축1spindleLoad 데이터')
        print(df['축1spindleLoad'].describe())
        plt.title(str(feedrate) + 'axis 1 spindleLoad histogram')
        plt.hist(df['축1spindleLoad'])
        plt.show()
        plt.title(str(feedrate) + 'axis 1 spindleLoad boxplot')
        plt.boxplot(df['축1spindleLoad'])
        plt.show()
        
        print('축1 spindleSpeed 데이터')
        print(df['축1 spindleSpeed'].describe())
        plt.title(str(feedrate) + 'axis 1 spindleSpeed histogram')
        plt.hist(df['축1 spindleSpeed'])
        plt.show()
        plt.title(str(feedrate) + 'axis 1 spindleSpeed boxplot')
        plt.boxplot(df['축1 spindleSpeed'])
        plt.show()
        
        print('Feed')
        print(df['Feed'].describe())
        plt.title(str(feedrate) + 'Feed histogram')
        plt.hist(df['Feed'])
        plt.show()
        plt.title(str(feedrate) + 'feed boxplot')
        plt.boxplot(df['Feed'])
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
        plt.show()
        
        plt.plot(df['축1 spindleSpeed'])
        plt.title(str(feedrate) + 'axis 1 spindleSpeed')
        plt.show()
        
        plt.plot(df['Feed'])
        plt.title(str(feedrate) + 'Feed')
        plt.show()
        
        return df
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
    
    df_2000 = DataFrame(data_2000, columns = ['축1spindleLoad', '축1 spindleSpeed', 'Feed'])
    df_2000 = data_anal.preprocess(df_2000, 2000)
    
    df_2200 = DataFrame(data_2200, columns = ['축1spindleLoad', '축1 spindleSpeed', 'Feed'])
    df_2200 = data_anal.preprocess(df_2200, 2200)
    
    df_2400 = DataFrame(data_2400, columns = ['축1spindleLoad', '축1 spindleSpeed', 'Feed'])
    df_2400 = data_anal.preprocess(df_2400, 2400)
    
    df_2600 = DataFrame(data_2600, columns = ['축1spindleLoad', '축1 spindleSpeed', 'Feed'])
    df_2600 = data_anal.preprocess(df_2600, 2600)
    
    
    