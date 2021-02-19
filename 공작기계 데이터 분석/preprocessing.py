import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os

def preprocess(df, Feed):
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
    
    Feed_outliers = df[(Feed-10) > df['Feed']].index
    df = df.drop(Feed_outliers)
    Feed_outliers = df[(Feed+10) < df['Feed']].index
    df = df.drop(Feed_outliers)
    # Feed_outliers = df[Feed != df['Feed']].index
    # df = df.drop(Feed_outliers)
    
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
    
    print(df.corr())
    sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2, 
                vmax=1, vmin=-1, fmt='1.2f')
    fig=plt.gcf()
    fig.set_size_inches(8,6)
    plt.title('Heat map')
    plt.show()
    
    plt.plot(df['축1spindleLoad'])
    plt.title('axis 1 spindleLoad')
    plt.show()
    
    plt.plot(df['축1 spindleSpeed'])
    plt.title('axis 1 spindleSpeed')
    plt.show()
    
    plt.plot(df['Feed'])
    plt.title('Feed')
    plt.show()
    
    return df

if __name__ == "__main__":
    data_types_logging ={'축1spindleLoad':'float64',
                         '축1 spidleSpeed':'float64',
                         'Feed':'float64'
                         }
    
    # 데이터 읽어오기
    df_logging = pd.read_csv('data/Feedrate 960.csv', dtype=data_types_logging,
                             engine='c', encoding='cp949')
    
    feedrate_960 = preprocess(df_logging, 960)